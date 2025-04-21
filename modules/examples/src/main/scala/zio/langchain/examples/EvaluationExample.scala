package zio.langchain.examples

import zio.*
import zio.Console.*

import zio.langchain.core.domain.*
import zio.langchain.core.model.LLM
import zio.langchain.core.retriever.Retriever
import zio.langchain.evaluation.*
import zio.langchain.integrations.openai.OpenAILLM

/**
 * Example demonstrating the use of the evaluation framework.
 * Shows how to evaluate LLM outputs and RAG system performance.
 */
object EvaluationExample extends ZIOAppDefault:

  /**
   * Main program that demonstrates various evaluation scenarios.
   */
  override def run: ZIO[Any, Throwable, Unit] =
    // Create an OpenAI LLM for both generation and evaluation
    val llm = OpenAILLM.default
    
    // Run the examples
    for
      _ <- printLine("=== ZIO LangChain Evaluation Examples ===")
      _ <- printLine("\n1. Basic LLM Output Evaluation")
      _ <- evaluateLLMOutput(llm)
      _ <- printLine("\n2. RAG System Evaluation")
      _ <- evaluateRAGSystem(llm)
      _ <- printLine("\n3. Comparative Evaluation")
      _ <- evaluateComparative(llm)
      _ <- printLine("\n4. Automated Metrics Evaluation")
      _ <- evaluateWithAutomatedMetrics()
    yield ()

  /**
   * Example of evaluating a basic LLM output.
   */
  private def evaluateLLMOutput(llm: LLM): ZIO[Any, Throwable, Unit] =
    // Define the input and expected criteria
    val input = "Explain the concept of quantum computing in simple terms."
    val criteria = Seq("accuracy", "clarity", "simplicity", "completeness")
    
    // Create a chain evaluator
    val evaluator = ChainEvaluator.make[String, String](
      llm,
      criteria,
      (input, output) => s"Input: $input\n\nOutput: $output"
    )
    
    // Generate a response using the LLM
    val generateResponse = llm.complete(input)
    
    // Evaluate the response
    for
      response <- generateResponse.tap(r => printLine(s"Generated response:\n$r\n"))
      result <- evaluator.evaluate(input, response)
      _ <- printLine("Evaluation result:")
      _ <- printLine(s"Overall score: ${result.score}")
      _ <- printLine("Individual metrics:")
      _ <- ZIO.foreach(result.metrics.toSeq.sortBy(_._1)) { case (metric, score) =>
        printLine(s"- $metric: $score")
      }
      _ <- ZIO.foreach(result.feedback) { feedback =>
        printLine(s"\nFeedback: $feedback")
      }
    yield ()

  /**
   * Example of evaluating a RAG system.
   */
  private def evaluateRAGSystem(llm: LLM): ZIO[Any, Throwable, Unit] =
    // Create a RAG evaluator
    val ragEvaluator = RAGEvaluator.make(llm)
    
    // Create a more detailed RAG metrics evaluator
    val metricsEvaluator = RAGMetricsEvaluator.make(llm)
    
    // Simulate a RAG system with mock data
    val query = "What are the environmental impacts of electric vehicles?"
    
    // Mock retrieved documents
    val retrievedDocs = Seq(
      Document(
        id = "doc1",
        content = """
          |Electric vehicles (EVs) have several environmental impacts. While they produce zero tailpipe emissions,
          |their overall environmental impact depends on how the electricity used to charge them is generated.
          |If the electricity comes from renewable sources like solar or wind, the environmental benefits are significant.
          |However, if it comes from coal or natural gas, the benefits are reduced.
        """.stripMargin.trim,
        metadata = Map("source" -> "environmental-journal")
      ),
      Document(
        id = "doc2",
        content = """
          |The production of EV batteries requires mining for materials like lithium, cobalt, and nickel,
          |which can have environmental impacts including habitat destruction, water usage, and potential pollution.
          |However, efforts are being made to improve battery recycling and reduce the need for new mining.
        """.stripMargin.trim,
        metadata = Map("source" -> "battery-technology-review")
      )
    )
    
    // Mock generated answer
    val generatedAnswer = """
      |Electric vehicles (EVs) have both positive and negative environmental impacts. On the positive side,
      |they produce zero tailpipe emissions, which helps reduce air pollution in urban areas. However, their
      |overall environmental impact depends on the source of electricity used for charging. EVs charged with
      |renewable energy have a much smaller carbon footprint than those charged using electricity from fossil fuels.
      |
      |The production of EV batteries also has environmental considerations. Mining for materials like lithium,
      |cobalt, and nickel can lead to habitat disruption and water usage concerns. The industry is working on
      |improving battery recycling programs and developing more sustainable mining practices to address these issues.
      |
      |Overall, studies suggest that EVs typically have a lower lifetime environmental impact than conventional
      |vehicles, especially as the electricity grid becomes cleaner.
    """.stripMargin.trim
    
    // Evaluate the RAG system
    for
      // Basic RAG evaluation
      basicResult <- ragEvaluator.evaluate(query, (retrievedDocs, generatedAnswer))
      _ <- printLine("Basic RAG Evaluation:")
      _ <- printLine(s"Overall score: ${basicResult.score}")
      _ <- printLine("Metrics:")
      _ <- ZIO.foreach(basicResult.metrics.toSeq.sortBy(_._1)) { case (metric, score) =>
        printLine(s"- $metric: $score")
      }
      
      // Detailed RAG metrics evaluation
      detailedResult <- metricsEvaluator.evaluate(query, retrievedDocs, generatedAnswer)
      _ <- printLine("\nDetailed RAG Metrics:")
      _ <- printLine(s"Context Relevance: ${detailedResult.contextRelevance}")
      _ <- printLine(s"Answer Relevance: ${detailedResult.answerRelevance}")
      _ <- printLine(s"Faithfulness: ${detailedResult.faithfulness}")
      _ <- printLine(s"Comprehensiveness: ${detailedResult.comprehensiveness}")
      _ <- printLine(s"Groundedness: ${detailedResult.groundedness}")
      _ <- printLine(s"Harmonious Score: ${detailedResult.harmoniousScore}")
    yield ()

  /**
   * Example of comparative evaluation between different systems.
   */
  private def evaluateComparative(llm: LLM): ZIO[Any, Throwable, Unit] =
    // Create a comparative evaluator
    val comparativeEvaluator = ComparativeEvaluator.make[String, String](
      llm,
      Seq("accuracy", "clarity", "completeness")
    )
    
    // Input query
    val query = "Explain how blockchain technology works."
    
    // Mock outputs from different systems
    val outputs = Seq(
      """
      |Blockchain is a distributed ledger technology that records transactions across many computers.
      |Each block contains a list of transactions and a hash of the previous block, forming a chain.
      |This structure makes the blockchain secure and resistant to modification.
      """.stripMargin.trim,
      
      """
      |Blockchain technology works by creating a decentralized database (or ledger) that is shared across a network
      |of computers (nodes). When a transaction occurs, it's grouped with others into a "block." Each block is verified
      |by the nodes through cryptographic algorithms, then added to the chain of previous blocks, creating a permanent,
      |immutable record. The key innovation is that this happens without a central authority - the network reaches
      |consensus about the state of the ledger through mechanisms like proof-of-work or proof-of-stake.
      """.stripMargin.trim
    )
    
    // Evaluate and compare the outputs
    for
      results <- comparativeEvaluator.evaluate(query, outputs)
      _ <- printLine("Comparative Evaluation Results:")
      _ <- ZIO.foreachDiscard(results.zipWithIndex) { case (result, index) =>
        for
          _ <- printLine(s"\nSystem ${index + 1} Score: ${result.score}")
          _ <- printLine("Metrics:")
          _ <- ZIO.foreach(result.metrics.toSeq.sortBy(_._1)) { case (metric, score) =>
            printLine(s"- $metric: $score")
          }
          _ <- ZIO.foreach(result.feedback) { feedback =>
            printLine(s"\nComparison: $feedback")
          }
        yield ()
      }
    yield ()

  /**
   * Example of evaluation using automated metrics without an LLM.
   */
  private def evaluateWithAutomatedMetrics(): ZIO[Any, Throwable, Unit] =
    // Create an automated metrics evaluator
    val automatedEvaluator = AutomatedMetricsEvaluator.make()
    
    // Reference (ground truth) and candidate texts
    val reference = """
      |Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence
      |displayed by animals including humans. AI research has been defined as the field of study of intelligent agents,
      |which refers to any system that perceives its environment and takes actions that maximize its chance of achieving
      |its goals.
    """.stripMargin.trim
    
    val candidate = """
      |Artificial intelligence (AI) is intelligence shown by machines, unlike the natural intelligence of humans and animals.
      |The field of AI research defines AI as the study of intelligent agents, which are systems that can understand their
      |surroundings and take actions to achieve specific goals.
    """.stripMargin.trim
    
    // Evaluate using all available automated metrics
    for
      result <- automatedEvaluator.evaluate(reference, candidate)
      _ <- printLine("Automated Metrics Evaluation:")
      _ <- printLine(s"Overall score: ${result.score}")
      _ <- printLine("Individual metrics:")
      _ <- ZIO.foreach(result.metrics.toSeq.sortBy(_._1)) { case (metric, score) =>
        printLine(s"- $metric: $score")
      }
    yield ()