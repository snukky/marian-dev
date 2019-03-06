#pragma once

#include "data/batch_generator.h"
#include "data/corpus.h"
#include "data/shortlist.h"
#include "data/text_input.h"

#include "3rd_party/threadpool.h"
#include "translator/history.h"
#include "translator/output_collector.h"
#include "translator/output_printer.h"

#include "models/model_task.h"
#include "translator/scorers.h"

namespace marian {

template <class Search>
class TranslateIterative : public ModelTask {
private:
  Ptr<Options> options_;
  std::vector<Ptr<ExpressionGraph>> graphs_;
  std::vector<std::vector<Ptr<Scorer>>> scorers_;

  Ptr<data::Corpus> corpus_;
  Ptr<Vocab> trgVocab_;
  Ptr<data::ShortlistGenerator> shortlistGenerator_;
  std::vector<Ptr<Vocab>> vocabs_;

  size_t numDevices_;

public:
  TranslateIterative(Ptr<Options> options) : options_(options) {
    options_->set("inference", true);
    // Iterative decoding requires line-by-line decoding at the moment
    options_->set<int>("mini-batch", 1);
    options_->set<int>("mini-batch-words", 0);

    corpus_ = New<data::Corpus>(options_, true);

    auto vocabs = options_->get<std::vector<std::string>>("vocabs");
    trgVocab_ = New<Vocab>(options_, vocabs.size() - 1);
    trgVocab_->load(vocabs.back());
    auto srcVocab = corpus_->getVocabs()[0];

    // TODO: this is bad!!!
    vocabs_.push_back(srcVocab);
    vocabs_.push_back(trgVocab_);

    if(options_->hasAndNotEmpty("shortlist"))
      shortlistGenerator_ = New<data::LexicalShortlistGenerator>(
          options_, srcVocab, trgVocab_, 0, 1, vocabs.front() == vocabs.back());

    auto devices = Config::getDevices(options_);
    numDevices_ = devices.size();

    ThreadPool threadPool(numDevices_, numDevices_);
    scorers_.resize(numDevices_);
    graphs_.resize(numDevices_);

    size_t id = 0;
    for(auto device : devices) {
      auto task = [&](DeviceId device, size_t id) {
        auto graph = New<ExpressionGraph>(true, options_->get<bool>("optimize"));
        graph->setDevice(device);
        graph->getBackend()->setClip(options_->get<float>("clip-gemm"));
        graph->reserveWorkspaceMB(options_->get<size_t>("workspace"));
        graphs_[id] = graph;

        auto scorers = createScorers(options_);
        for(auto scorer : scorers) {
          scorer->init(graph);
          if(shortlistGenerator_)
            scorer->setShortlistGenerator(shortlistGenerator_);
        }

        scorers_[id] = scorers;
        graph->forward();
      };

      threadPool.enqueue(task, device, id++);
    }
  }

  void run() override {
    data::BatchGenerator<data::Corpus> bg(corpus_, options_);

    ThreadPool threadPool(numDevices_, numDevices_);
    size_t batchId = 0;

    auto optionsPrint = New<Options>(options_->clone());
    optionsPrint->set("n-best", true);
    auto printer = New<OutputPrinter>(optionsPrint, trgVocab_);

    auto collector = New<OutputCollector>(options_->get<std::string>("output"));
    if(options_->get<bool>("quiet-translation"))
      collector->setPrintingStrategy(New<QuietPrinting>());

    bg.prepare(false);

    auto maxIter = options_->get<size_t>("iterative-max");
    auto threshold = options_->get<float>("iterative-threshold");
    auto debug = options_->get<bool>("iterative-debug");

    for(auto batch : bg) {
      auto task = [=](size_t id) {
        thread_local Ptr<ExpressionGraph> graph;
        thread_local std::vector<Ptr<Scorer>> scorers;

        if(!graph) {
          graph = graphs_[id % numDevices_];
          scorers = scorers_[id % numDevices_];
        }

        std::string inputNext = trgVocab_->decode(batch->front()->data());

        if(debug)
          std::cerr << "Original input: " << inputNext << std::endl;

        size_t iter = 0;

        std::string output;
        Ptr<History> outputHistory;

        do {
          if(debug) {
            std::cerr << "== Iteration " << iter + 1 << "/" << maxIter << " ==" << std::endl
                      << "Input: " << inputNext << std::endl;
          }

          auto inputText = New<data::TextInput>(std::vector<std::string>({inputNext}), vocabs_, options_);
          auto inputGen = New<data::BatchGenerator<data::TextInput>>(inputText, options_);
          inputGen->prepare(false);
          auto search = New<Search>(options_, scorers, trgVocab_->getEosId(), trgVocab_->getUnkId());
          auto history = search->search(graph, *inputGen->begin())[0];

          float costId = -std::numeric_limits<float>::max();
          float costNonId = -std::numeric_limits<float>::max();
          std::string inputNonId;
          Ptr<History> historyNonId;

          auto nbestlist = printer->print(history);
          for(auto const& transWithScore : nbestlist) {
            const std::string& trans = transWithScore.first;
            const float& score = transWithScore.second;

            if(debug)
              std::cerr << trans << "\t" << score;

            if(trans == inputNext) {
              if(debug)
                std::cerr << "\t<- identity";
              costId = score;
            } else if(costNonId < score) {
              if(debug)
                std::cerr << "\t<- best non-identity";
              costNonId = score;
              inputNonId = trans;
              historyNonId = history;
            }
            if(debug)
              std::cerr << std::endl;
          }

          if(debug)
            std::cerr << "Checking if costNonId / costId < " << threshold << " : "
                      << costNonId / costId << std::endl;

          if((costNonId / costId) < threshold) {
            output = inputNonId;
            outputHistory = historyNonId;
            if(debug)
              std::cerr << "Rewriting best output: " << output << std::endl;
          } else {
            output = inputNext;
            outputHistory = history;
          }

          if(inputNext == output)
            break;
          inputNext = output;

          ++iter;
        } while(iter < maxIter);

        if(debug)
          std::cerr << "Final output: " << output << std::endl;

        std::stringstream best1;
        std::stringstream bestn;
        printer->print(outputHistory, best1, bestn);
        collector->Write(id, best1.str(), bestn.str(), options_->get<bool>("n-best"));
      };

      threadPool.enqueue(task, batchId++);
    }
  }
};
}  // namespace marian
