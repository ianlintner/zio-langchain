package zio.langchain.examples

import zio.*
import zio.Console.*
import zio.http.Client

import zio.langchain.core.model.*
import zio.langchain.core.document.*
import zio.langchain.core.domain.*
import zio.langchain.core.retriever.*
import zio.langchain.core.errors.RetrieverError
import zio.langchain.integrations.openai.*
import zio.langchain.rag.*

/**
 * Example demonstrating query transformation techniques for improved RAG performance.
 *
 * This example shows how to:
 * 1. Use different query transformation techniques (expansion, HyDE, multi-query)
 * 2. Create transforming retrievers that apply these techniques
 * 3. Compare retrieval results with and without query transformation
 *
 * To run this example:
 * 1. Set your OpenAI API key in the environment variable OPENAI_API_KEY
 * 2. Run the example using: `sbt "examples/runMain zio.langchain.examples.QueryTransformationExample"`
 */
object QueryTransformationExample extends ZIOAppDefault {

  // Sample documents for our vector store
  val documents = Seq(
    Document(
      id = "doc1",
      content = "ZIO is a library for asynchronous and concurrent programming in Scala.",
      metadata = Map("source" -> "docs", "category" -> "programming")
    ),
    Document(
      id = "doc2",
      content = "ZIO provides a powerful effect system for functional programming.",
      metadata = Map("source" -> "docs", "category" -> "programming")
    ),
    Document(
      id = "doc3",
      content = "ZIO LangChain is a Scala library for working with Large Language Models.",
      metadata = Map("source" -> "docs", "category" -> "ai")
    ),
    Document(
      id = "doc4",
      content = "Retrieval-Augmented Generation (RAG) combines retrieval systems with language generation.",
      metadata = Map("source" -> "docs", "category" -> "ai")
    ),
    Document(
      id = "doc5",
      content = "Query transformation techniques can improve RAG performance by reformulating user queries.",
      metadata = Map("source" -> "docs", "category" -> "ai")
    ),
    Document(
      id = "doc6",
      content = "Functional programming uses immutable data structures and pure functions.",
      metadata = Map("source" -> "docs", "category" -> "programming")
    ),
    Document(
      id = "doc7",
      content = "Hypothetical Document Embeddings (HyDE) generates synthetic documents to improve retrieval.",
      metadata = Map("source" -> "docs", "category" -> "ai")
    )
  )

  /**
   * A simple in-memory retriever for demonstration purposes
   */
  class SimpleInMemoryRetriever(
    documents: Seq[Document],
    embeddings: Seq[(Document, Embedding)]
  ) extends zio.langchain.core.retriever.Retriever {
    override def retrieve(query: String, maxResults: Int = 10): ZIO[Any, zio.langchain.core.errors.RetrieverError, Seq[Document]] = {
      retrieveWithScores(query, maxResults).map(_.map(_._1))
    }

    override def retrieveWithScores(query: String, maxResults: Int = 10): ZIO[Any, zio.langchain.core.errors.RetrieverError, Seq[(Document, Double)]] = {
      // Just simulate retrieval based on random scoring for this example
      ZIO.succeed {
        // Create simulated similarity scores (random for demonstration)
        val rnd = new scala.util.Random(query.hashCode)
        val similarities = embeddings.map { case (doc, _) =>
          // Generate a random similarity score between 0 and 1
          val similarityScore = 0.5 + rnd.nextDouble() * 0.5
          (doc, similarityScore)
        }
        
        // Sort by similarity (highest first) and take top results
        similarities.sortBy { case (_, score) => -score }.take(maxResults)
      }
    }
  }

  /**
   * Retrieve documents with the base retriever (no query transformation)
   */
  def retrieveBaseline(
    query: String,
    retriever: Retriever
  ): ZIO[Any, Throwable, Seq[(Document, Double)]] = {
    for {
      _ <- printLine(s"[Baseline] Query: '$query'")
      results <- retriever.retrieveWithScores(query)
      _ <- ZIO.foreach(results) { case (doc, score) =>
        printLine(s"[Baseline] Retrieved (score: ${String.format("%.4f", score)}): ${doc.content}")
      }
    } yield results
  }

  /**
   * Retrieve documents with query expansion transformation
   */
  def retrieveWithExpansion(
    query: String,
    retriever: Retriever,
    llm: LLM
  ): ZIO[Any, Throwable, Seq[(Document, Double)]] = {
    for {
      _ <- printLine(s"\n[Query Expansion] Original query: '$query'")
      
      // Create query expansion transformer and transforming retriever
      transformer = new QueryExpansionTransformer(llm)
      transformingRetriever = new TransformingRetriever(retriever, transformer)
      
      // Transform the query for logging
      expandedQuery <- transformer.transform(query)
      _ <- printLine(s"[Query Expansion] Expanded query: '$expandedQuery'")
      
      // Retrieve documents
      results <- transformingRetriever.retrieveWithScores(query)
      _ <- ZIO.foreach(results) { case (doc, score) =>
        printLine(s"[Query Expansion] Retrieved (score: ${String.format("%.4f", score)}): ${doc.content}")
      }
    } yield results
  }

  /**
   * Retrieve documents with Hypothetical Document Embeddings (HyDE) transformation
   */
  def retrieveWithHyDE(
    query: String,
    retriever: Retriever,
    llm: LLM
  ): ZIO[Any, Throwable, Seq[(Document, Double)]] = {
    for {
      _ <- printLine(s"\n[HyDE] Original query: '$query'")
      
      // Create HyDE transformer and transforming retriever
      transformer = new HyDETransformer(llm)
      transformingRetriever = new TransformingRetriever(retriever, transformer)
      
      // Transform the query for logging
      hydeDoc <- transformer.transform(query)
      _ <- printLine(s"[HyDE] Generated hypothetical document: '$hydeDoc'")
      
      // Retrieve documents
      results <- transformingRetriever.retrieveWithScores(query)
      _ <- ZIO.foreach(results) { case (doc, score) =>
        printLine(s"[HyDE] Retrieved (score: ${String.format("%.4f", score)}): ${doc.content}")
      }
    } yield results
  }

  /**
   * Retrieve documents with Multi-query transformation
   */
  def retrieveWithMultiQuery(
    query: String,
    retriever: Retriever,
    llm: LLM
  ): ZIO[Any, Throwable, Seq[(Document, Double)]] = {
    for {
      _ <- printLine(s"\n[Multi-Query] Original query: '$query'")
      
      // Create multi-query transformer and transforming retriever
      transformer = new MultiQueryTransformer(llm, 3)
      transformingRetriever = new TransformingRetriever(retriever, transformer)
      
      // Transform the query for logging
      queries <- transformer.transform(query)
      _ <- printLine(s"[Multi-Query] Generated queries: '${queries.split('\n').mkString("', '")}'")
      
      // Retrieve documents
      results <- transformingRetriever.retrieveWithScores(query)
      _ <- ZIO.foreach(results) { case (doc, score) =>
        printLine(s"[Multi-Query] Retrieved (score: ${String.format("%.4f", score)}): ${doc.content}")
      }
    } yield results
  }

  /**
   * Retrieve documents with all transformations combined
   */
  def retrieveWithCombinedTransformations(
    query: String,
    retriever: Retriever,
    llm: LLM
  ): ZIO[Any, Throwable, Seq[(Document, Double)]] = {
    for {
      _ <- printLine(s"\n[Combined] Original query: '$query'")
      
      // Create different transformers
      expansionTransformer = new QueryExpansionTransformer(llm)
      hydeTransformer = new HyDETransformer(llm)
      multiQueryTransformer = new MultiQueryTransformer(llm, 2)
      
      // Create multi-transforming retriever with all transformers
      multiTransformingRetriever = new MultiTransformingRetriever(
        retriever,
        Seq(expansionTransformer, hydeTransformer, multiQueryTransformer),
        deduplicateResults = true
      )
      
      // Retrieve documents
      results <- multiTransformingRetriever.retrieveWithScores(query)
      _ <- printLine(s"[Combined] Retrieved ${results.size} documents with combined transformations:")
      _ <- ZIO.foreach(results) { case (doc, score) =>
        printLine(s"[Combined] Retrieved (score: ${String.format("%.4f", score)}): ${doc.content}")
      }
    } yield results
  }

  /**
   * Compare the most relevant document retrieval between different transformation methods
   */
  def compareResults(
    baselineResults: Seq[(Document, Double)],
    expansionResults: Seq[(Document, Double)],
    hydeResults: Seq[(Document, Double)],
    multiQueryResults: Seq[(Document, Double)],
    combinedResults: Seq[(Document, Double)]
  ): ZIO[Any, Nothing, Unit] = {
    val collectTopDocIds = (results: Seq[(Document, Double)]) => 
      results.map(_._1.id).take(3).toSet
    
    val baselineTopDocs = collectTopDocIds(baselineResults)
    val expansionTopDocs = collectTopDocIds(expansionResults)
    val hydeTopDocs = collectTopDocIds(hydeResults)
    val multiQueryTopDocs = collectTopDocIds(multiQueryResults)
    val combinedTopDocs = collectTopDocIds(combinedResults)
    
    for {
      _ <- printLine("\n=== Comparison of Top Results ===").orDie
      _ <- printLine(s"Baseline top docs: ${baselineTopDocs.mkString(", ")}").orDie
      _ <- printLine(s"Query Expansion top docs: ${expansionTopDocs.mkString(", ")}").orDie
      _ <- printLine(s"HyDE top docs: ${hydeTopDocs.mkString(", ")}").orDie
      _ <- printLine(s"Multi-Query top docs: ${multiQueryTopDocs.mkString(", ")}").orDie
      _ <- printLine(s"Combined top docs: ${combinedTopDocs.mkString(", ")}").orDie
      
      // Compare each transformation method with baseline
      _ <- printLine("\n=== Differences from Baseline ===").orDie
      _ <- printLine(s"Query Expansion unique docs: ${(expansionTopDocs -- baselineTopDocs).mkString(", ")}").orDie
      _ <- printLine(s"HyDE unique docs: ${(hydeTopDocs -- baselineTopDocs).mkString(", ")}").orDie
      _ <- printLine(s"Multi-Query unique docs: ${(multiQueryTopDocs -- baselineTopDocs).mkString(", ")}").orDie
      _ <- printLine(s"Combined unique docs: ${(combinedTopDocs -- baselineTopDocs).mkString(", ")}").orDie
    } yield ()
  }

  override def run = {
    // Define our program
    val program = for {
      // Print welcome message
      _ <- printLine("Welcome to ZIO LangChain Query Transformation Example!").orDie
      
      // Get the models
      embeddingModel <- ZIO.service[EmbeddingModel]
      llm <- ZIO.service[LLM]
      
      // Create embeddings for documents
      _ <- printLine("Creating embeddings for sample documents...").orDie
      embeddedDocs <- embeddingModel.embedDocuments(documents)
      _ <- printLine(s"Created embeddings for ${embeddedDocs.size} documents.").orDie
      
      // Create the simple in-memory retriever
      retriever = new SimpleInMemoryRetriever(documents, embeddedDocs)
      
      // First query
      query1 = "How does ZIO help with programming?"
      _ <- printLine(s"\n=== Query: '$query1' ===").orDie
      _ <- printLine(s"\n=== Query: '$query1' ===")
      
      // Run different retrieval methods
      baseline1 <- retrieveBaseline(query1, retriever)
      expansion1 <- retrieveWithExpansion(query1, retriever, llm)
      hyde1 <- retrieveWithHyDE(query1, retriever, llm)
      multiQuery1 <- retrieveWithMultiQuery(query1, retriever, llm)
      combined1 <- retrieveWithCombinedTransformations(query1, retriever, llm)
      
      // Compare results
      _ <- compareResults(baseline1, expansion1, hyde1, multiQuery1, combined1)
      
      // Try another query
      query2 = "Tell me about RAG systems"
      _ <- printLine(s"\n\n=== Query: '$query2' ===")
      
      // Run different retrieval methods for second query
      baseline2 <- retrieveBaseline(query2, retriever)
      expansion2 <- retrieveWithExpansion(query2, retriever, llm)
      hyde2 <- retrieveWithHyDE(query2, retriever, llm)
      multiQuery2 <- retrieveWithMultiQuery(query2, retriever, llm)
      combined2 <- retrieveWithCombinedTransformations(query2, retriever, llm)
      
      // Compare results
      _ <- compareResults(baseline2, expansion2, hyde2, multiQuery2, combined2)
      
      // Final message
      _ <- printLine("\nQuery Transformation example completed successfully.")
    } yield ()
    
    // Provide dependencies and run the program
    program.provide(
      // HTTP Client dependency required by all API integrations
      Client.default,
      
      // OpenAI LLM layer
      OpenAILLM.live,
      
      // OpenAI embedding model layer
      OpenAIEmbedding.live,
      
      // OpenAI LLM configuration
      ZLayer.succeed(
        OpenAIConfig(
          apiKey = sys.env.getOrElse("OPENAI_API_KEY", ""),
          model = "gpt-3.5-turbo"
        )
      ),
      
      // OpenAI embedding configuration
      ZLayer.succeed(
        OpenAIEmbeddingConfig(
          apiKey = sys.env.getOrElse("OPENAI_API_KEY", ""),
          model = "text-embedding-ada-002"
        )
      )
    )
  }
}