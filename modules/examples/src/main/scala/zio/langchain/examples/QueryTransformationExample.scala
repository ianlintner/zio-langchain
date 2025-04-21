package zio.langchain.examples

import zio.*
import zio.Console.*
import zio.stream.ZStream

import zio.langchain.core.model.*
import zio.langchain.core.retriever.*
import zio.langchain.core.document.*
import zio.langchain.core.domain.*
import zio.langchain.core.chain.*
import zio.langchain.core.errors.*
import zio.langchain.integrations.openai.*
import zio.langchain.rag.*

import java.nio.file.{Path, Files}

/**
 * An example demonstrating advanced RAG techniques with query transformation.
 * This example shows how to:
 * 1. Create and use different query transformation strategies
 * 2. Integrate query transformers with retrievers
 * 3. Compare retrieval results with and without query transformation
 * 4. Use multiple query transformations for improved retrieval
 */
object QueryTransformationExample extends ZIOAppDefault:
  /**
   * The main program.
   */
  override def run: ZIO[Any, Throwable, Unit] =
    // Create the program
    val program = for
      // Print welcome message
      _ <- ZIO.logInfo("Welcome to ZIO LangChain Query Transformation Example!")
      
      // Get the services
      llm <- ZIO.service[LLM]
      embeddingModel <- ZIO.service[EmbeddingModel]
      
      // Create sample documents
      documents = createSampleDocuments()
      _ <- ZIO.logInfo(s"Created ${documents.size} sample documents")
      
      // Create embeddings for the documents
      _ <- ZIO.logInfo("Creating embeddings for documents...")
      embeddedDocs <- embeddingModel.embedDocuments(documents)
      _ <- ZIO.logInfo("Created embeddings for all documents")
      
      // Create a basic retriever (without query transformation)
      basicRetriever = new VectorRetriever(embeddedDocs, embeddingModel)
      
      // Create query transformers
      _ <- ZIO.logInfo("Creating query transformers...")
      queryExpansion <- ZIO.service[QueryTransformer].provide(queryExpansionTransformer)
      hyde <- ZIO.service[QueryTransformer].provide(hydeTransformer)
      multiQuery <- ZIO.service[QueryTransformer].provide(multiQueryTransformer(2))
      
      // Create transforming retrievers
      expansionRetriever = TransformingRetriever(basicRetriever, queryExpansion)
      hydeRetriever = TransformingRetriever(basicRetriever, hyde)
      multiQueryRetriever = MultiTransformingRetriever(basicRetriever, Seq(queryExpansion, hyde))
      
      // Run the demo
      _ <- runDemo(basicRetriever, expansionRetriever, hydeRetriever, multiQueryRetriever)
    yield ()
    
    // Provide the required services and run the program
    program.provide(
      // OpenAI LLM layer
      OpenAILLM.live,
      // OpenAI LLM configuration layer
      ZLayer.succeed(
        OpenAIConfig(
          apiKey = sys.env.getOrElse("OPENAI_API_KEY", ""),
          model = "gpt-4o", // Using a capable model for query transformation
          temperature = 0.0 // Low temperature for deterministic responses
        )
      ),
      // OpenAI Embedding layer
      OpenAIEmbedding.live,
      // OpenAI Embedding configuration layer
      ZLayer.succeed(
        OpenAIEmbeddingConfig(
          apiKey = sys.env.getOrElse("OPENAI_API_KEY", ""),
          model = "text-embedding-3-large" // Using the latest embedding model
        )
      )
    )
  
  /**
   * Creates sample documents for demonstration purposes.
   *
   * @return A sequence of sample documents
   */
  private def createSampleDocuments(): Seq[Document] =
    Seq(
      Document(
        id = "doc1",
        content = """ZIO is a library for asynchronous and concurrent programming in Scala. 
          |ZIO provides a simple, type-safe, testable, and performant foundation for 
          |building asynchronous and concurrent applications. At its core, ZIO is built around 
          |an effect data type that encapsulates all the ways a program can fail, 
          |interact with the external world, or return a value.""".stripMargin,
        metadata = Map("source" -> "zio-overview.txt", "topic" -> "programming")
      ),
      Document(
        id = "doc2",
        content = """LangChain is a framework for developing applications powered by language models. 
          |It enables applications that are context-aware, reason, and learn from feedback. 
          |LangChain provides standard interfaces for LLMs, prompt templates, chains, and agents, 
          |making it easy to swap components and reuse code across different applications.""".stripMargin,
        metadata = Map("source" -> "langchain-intro.txt", "topic" -> "ai")
      ),
      Document(
        id = "doc3",
        content = """Scala is a high-level programming language that combines object-oriented and 
          |functional programming paradigms. It was designed to be concise, elegant, and type-safe. 
          |Scala runs on the JVM and integrates seamlessly with Java libraries. Its type system supports 
          |both static type-checking and type inference, helping catch errors early while reducing verbosity.""".stripMargin,
        metadata = Map("source" -> "scala-intro.txt", "topic" -> "programming")
      ),
      Document(
        id = "doc4",
        content = """Retrieval-Augmented Generation (RAG) is a technique that enhances Large Language Models 
          |by retrieving relevant information from external knowledge sources before generating responses. 
          |This approach helps ground the model's outputs in factual, up-to-date information, reducing 
          |hallucinations and improving accuracy. In a typical RAG system, user queries are used to search 
          |and retrieve relevant documents, which are then provided as context to the LLM.""".stripMargin,
        metadata = Map("source" -> "rag-explanation.txt", "topic" -> "ai")
      ),
      Document(
        id = "doc5",
        content = """Embeddings are numerical representations of text that capture semantic meaning. 
          |They convert words, phrases, or documents into dense vectors in a high-dimensional space, 
          |where semantically similar text has similar vector representations. This enables powerful 
          |applications like semantic search, where documents are retrieved based on meaning rather than 
          |just keyword matching. Modern embedding models like those from OpenAI can capture nuanced 
          |relationships between different pieces of text.""".stripMargin,
        metadata = Map("source" -> "embeddings-explained.txt", "topic" -> "ai")
      ),
      Document(
        id = "doc6",
        content = """Query transformation is an advanced RAG technique that improves retrieval performance
          |by modifying the original user query. Common approaches include query expansion (adding related terms),
          |Hypothetical Document Embeddings (HyDE), which generates a hypothetical answer to use as the query,
          |and multi-query generation, which creates multiple variations of the query to capture different aspects.
          |These techniques help bridge the semantic gap between user questions and relevant documents.""".stripMargin,
        metadata = Map("source" -> "query-transformation.txt", "topic" -> "ai")
      ),
      Document(
        id = "doc7",
        content = """ZIO offers a comprehensive ecosystem of libraries built on top of its core functionality.
          |These include ZIO Streams for streaming data processing, ZIO HTTP for building HTTP servers and clients,
          |ZIO JSON for JSON encoding/decoding, ZIO Config for configuration management, ZIO Cache for caching,
          |ZIO Logging for structured logging, and many more. These libraries are designed to work seamlessly
          |together, providing a consistent and powerful platform for building applications.""".stripMargin,
        metadata = Map("source" -> "zio-ecosystem.txt", "topic" -> "programming")
      ),
      Document(
        id = "doc8",
        content = """The Scala programming language was created by Martin Odersky and first released in 2004.
          |Scala's name is a portmanteau of "scalable" and "language," highlighting its design goal of growing
          |with the needs of its users. Scala 3, released in 2021, introduced significant improvements to the
          |language, including a new syntax, improved type inference, union types, intersection types, and
          |opaque type aliases. Scala is used by companies like Twitter, Netflix, and LinkedIn for their
          |critical infrastructure.""".stripMargin,
        metadata = Map("source" -> "scala-history.txt", "topic" -> "programming")
      )
    )
  
  /**
   * A simple vector-based retriever for demonstration purposes.
   *
   * @param documents The documents with their embeddings
   * @param embeddingModel The embedding model to use for query embedding
   */
  private class VectorRetriever(
    documents: Seq[(Document, Embedding)],
    embeddingModel: EmbeddingModel
  ) extends Retriever:
    override def retrieve(query: String, maxResults: Int): ZIO[Any, RetrieverError, Seq[Document]] =
      retrieveWithScores(query, maxResults).map(_.map(_._1))
    
    override def retrieveWithScores(query: String, maxResults: Int): ZIO[Any, RetrieverError, Seq[(Document, Double)]] =
      embeddingModel.embedQuery(query)
        .mapError(e => RetrieverError(e, s"Failed to embed query: ${e.getMessage}"))
        .map { queryEmbedding =>
          documents.map { case (doc, embedding) =>
            val score = queryEmbedding.cosineSimilarity(embedding).toDouble
            (doc, score)
          }
          .sortBy(-_._2)
          .take(maxResults)
        }
  
  /**
   * Runs the demonstration, comparing different retrieval approaches.
   *
   * @param basicRetriever The basic retriever without query transformation
   * @param expansionRetriever The retriever with query expansion
   * @param hydeRetriever The retriever with HyDE
   * @param multiQueryRetriever The retriever with multiple query transformations
   */
  private def runDemo(
    basicRetriever: Retriever,
    expansionRetriever: Retriever,
    hydeRetriever: Retriever,
    multiQueryRetriever: Retriever
  ): ZIO[Any, Throwable, Unit] =
    val queries = Seq(
      "How does ZIO help with concurrent programming?",
      "What are embeddings used for in AI?",
      "Tell me about functional programming in Scala"
    )
    
    ZIO.foreach(queries) { query =>
      for
        _ <- ZIO.logInfo(s"\n\n=== Testing query: '$query' ===\n")
        
        // Basic retrieval
        _ <- ZIO.logInfo("--- Basic Retrieval (No Transformation) ---")
        basicResults <- basicRetriever.retrieve(query, 2)
        _ <- ZIO.foreach(basicResults) { doc =>
          ZIO.logInfo(s"Document: ${doc.id} (${doc.metadata.getOrElse("source", "unknown")})")
          ZIO.logInfo(s"Content: ${doc.content.take(100)}...")
        }
        
        // Query expansion
        _ <- ZIO.logInfo("\n--- Query Expansion Transformation ---")
        expansionResults <- expansionRetriever.retrieve(query, 2)
        _ <- ZIO.foreach(expansionResults) { doc =>
          ZIO.logInfo(s"Document: ${doc.id} (${doc.metadata.getOrElse("source", "unknown")})")
          ZIO.logInfo(s"Content: ${doc.content.take(100)}...")
        }
        
        // HyDE
        _ <- ZIO.logInfo("\n--- Hypothetical Document Embeddings (HyDE) Transformation ---")
        hydeResults <- hydeRetriever.retrieve(query, 2)
        _ <- ZIO.foreach(hydeResults) { doc =>
          ZIO.logInfo(s"Document: ${doc.id} (${doc.metadata.getOrElse("source", "unknown")})")
          ZIO.logInfo(s"Content: ${doc.content.take(100)}...")
        }
        
        // Multi-query
        _ <- ZIO.logInfo("\n--- Multi-Query Transformation ---")
        multiResults <- multiQueryRetriever.retrieve(query, 2)
        _ <- ZIO.foreach(multiResults) { doc =>
          ZIO.logInfo(s"Document: ${doc.id} (${doc.metadata.getOrElse("source", "unknown")})")
          ZIO.logInfo(s"Content: ${doc.content.take(100)}...")
        }
      yield ()
    }