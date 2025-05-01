package zio.langchain.examples

import zio.*
import zio.Console.*
import zio.http.Client

import zio.langchain.core.model.*
import zio.langchain.core.document.*
import zio.langchain.core.domain.*
import zio.langchain.core.memory.Memory
import zio.langchain.core.retriever.Retriever
import zio.langchain.core.errors.RetrieverError
import zio.langchain.document.chunkers.{DocumentChunker, DocumentChunkers}
import zio.langchain.integrations.openai.*
import zio.langchain.memory.BufferMemory

/**
 * Example demonstrating an enhanced Retrieval-Augmented Generation (RAG) system.
 *
 * This example builds on SimpleRAG with additional features:
 * 1. Document chunking for better context handling
 * 2. Memory integration to track conversation history
 * 3. Smarter prompt engineering
 */
object EnhancedRAG extends ZIOAppDefault {

  // Sample documents - more extensive than SimpleRAG
  val documents = Seq(
    Document(
      id = "doc1",
      content = "ZIO is a library for asynchronous and concurrent programming in Scala. " +
                "It provides a pure functional approach to handling effects like IO, " +
                "making it easier to write concurrent, resilient, and efficient applications. " +
                "ZIO offers features like fiber-based concurrency, structured error handling, " +
                "and resource-safe programming.",
      metadata = Map("source" -> "zio-docs", "category" -> "overview")
    ),
    Document(
      id = "doc2",
      content = "ZIO provides a powerful effect system for functional programming. " +
                "Effects in ZIO are values that describe computations that may fail, " +
                "interact with the external world, or never terminate. ZIO's effect system " +
                "allows for composing effects in a pure functional way, without sacrificing " +
                "performance or type safety.",
      metadata = Map("source" -> "zio-docs", "category" -> "effects")
    ),
    Document(
      id = "doc3",
      content = "ZIO's concurrency model is based on fibers, which are lightweight " +
                "threads managed by ZIO runtime. Fibers are much more efficient than " +
                "JVM threads and allow for writing highly concurrent applications without " +
                "the complexity of traditional multithreading. Fibers can be forked, joined, " +
                "interrupted, and composed with ease.",
      metadata = Map("source" -> "zio-docs", "category" -> "concurrency")
    ),
    Document(
      id = "doc4",
      content = "ZIO LangChain is a Scala wrapper around the LangChain concepts with ZIO integration. " +
                "It provides a type-safe and functional approach to building LLM-powered applications. " +
                "The library includes modules for working with various LLMs, embedding models, " +
                "vector stores, and RAG systems.",
      metadata = Map("source" -> "langchain-docs", "category" -> "overview")
    ),
    Document(
      id = "doc5",
      content = "Retrieval-Augmented Generation (RAG) is a technique that enhances LLM responses " +
                "by retrieving relevant context from a knowledge base before generating answers. " +
                "This approach grounds LLM outputs in factual information, reducing hallucinations " +
                "and improving accuracy for domain-specific tasks.",
      metadata = Map("source" -> "langchain-docs", "category" -> "rag")
    )
  )

  /**
   * Helper function to transform queries for better retrieval.
   */
  def enhanceQuery(query: String): String = {
    if (query.toLowerCase.contains("what is") || query.toLowerCase.contains("explain")) {
      // For definitional queries, extract the concept being asked about
      s"concept definition: ${query.replaceAll("(?i)what is|explain", "").trim}"
    } else if (query.toLowerCase.contains("how to")) {
      // For procedural queries, focus on the task
      s"process tutorial: ${query.replaceAll("(?i)how to", "").trim}"
    } else if (query.toLowerCase.contains("difference between")) {
      // For comparative queries, highlight the comparison
      s"comparison: ${query.trim}"
    } else {
      // Default enhancement adds search-friendly terms
      s"information about: ${query.trim}"
    }
  }

  /**
   * Create an enhanced retriever that uses document chunking.
   */
  def createEnhancedRetriever(
    embeddedDocs: Seq[(Document, Embedding)],
    embeddingModel: EmbeddingModel
  ): Retriever = new Retriever {
    
    override def retrieve(
      query: String, 
      maxResults: Int = 3
    ): ZIO[Any, RetrieverError, Seq[Document]] = {
      val transformedQuery = enhanceQuery(query)
      
      embeddingModel.embed(transformedQuery)
        .mapError(e => RetrieverError(e))
        .map { queryEmbedding =>
          val similarityScores = embeddedDocs.map { case (doc, embedding) =>
            (doc, queryEmbedding.cosineSimilarity(embedding))
          }
          
          similarityScores.sortBy(-_._2).take(maxResults).map(_._1)
        }
    }
  }

  /**
   * Generate a response using RAG with memory integration.
   */
  def generateResponse(
    query: String,
    memory: Memory,
    retriever: Retriever,
    llm: LLM
  ): ZIO[Any, Throwable, String] = {
    for {
      // Retrieve conversation history
      chatHistory <- memory.get
      
      // Get relevant documents
      relevantDocs <- retriever.retrieve(query)
        .mapError(e => new RuntimeException(s"Retrieval error: ${e.message}", e))
      
      // Format context from documents
      contextText = relevantDocs.map { doc =>
        s"""Source: ${doc.metadata.getOrElse("source", "unknown")}
           |Category: ${doc.metadata.getOrElse("category", "unknown")}
           |Content: ${doc.content}
           |""".stripMargin
      }.mkString("\n")
      
      // Create a system message with instructions
      systemMessage = ChatMessage.system(
        """You are a helpful assistant specializing in ZIO and functional programming.
          |When answering questions, use the provided context information.
          |If the context doesn't contain relevant information, say that you don't have enough information.
          |Always maintain a technical but approachable tone.
          |""".stripMargin
      )
      
      // Create a context message with retrieved information
      contextMessage = ChatMessage.system(
        s"""Here is information relevant to the user's question:
           |
           |$contextText
           |
           |Use this information to provide an accurate and helpful response.
           |""".stripMargin
      )
      
      // Add the user's current query
      userMessage = ChatMessage.user(query)
      
      // Build the complete message list
      messages = Seq(systemMessage, contextMessage) ++ 
                 chatHistory.filter(m => m.role != Role.System) ++ 
                 Seq(userMessage)
      
      // Generate response
      response <- llm.completeChat(messages)
      
      // Add both user query and assistant response to memory
      _ <- memory.add(userMessage)
      _ <- memory.add(response.message)
    } yield response.message.contentAsString
  }

  override def run = {
    // Define our program
    val program = for {
      // Initialize components
      _ <- printLine("Welcome to ZIO LangChain Enhanced RAG Example!")
      
      // Get required services
      embeddingModel <- ZIO.service[EmbeddingModel]
      llm <- ZIO.service[LLM]
      memory <- ZIO.service[Memory]
      
      // Process and embed documents
      _ <- printLine("Creating embeddings for sample documents...")
      embeddedDocs <- embeddingModel.embedDocuments(documents)
      _ <- printLine("Embeddings created successfully.")
      
      // Create enhanced retriever
      retriever = createEnhancedRetriever(embeddedDocs, embeddingModel)
      
      // Interactive chat loop
      _ <- printLine("\nYou can now ask questions about ZIO (type 'exit' to quit):")
      _ <- (for {
        query <- readLine
        _ <- ZIO.when(query.trim.toLowerCase != "exit") {
          for {
            // Process the query and generate a response
            response <- generateResponse(query, memory, retriever, llm)
            
            // Display the response
            _ <- printLine(s"\nAssistant: $response\n")
            _ <- printLine("Ask another question or type 'exit' to quit:")
          } yield ()
        }
      } yield query).repeatWhile(query => query.trim.toLowerCase != "exit")
      
      // Final message
      _ <- printLine("\nThank you for using the Enhanced RAG example!")
    } yield ()
    
    // Provide all required dependencies
    program.provide(
      // HTTP Client dependency required by all API integrations
      Client.default,
      
      // OpenAI LLM layer
      OpenAILLM.live,
      
      // OpenAI embedding model layer
      OpenAIEmbedding.live,
      
      // Memory layer for conversation history
      BufferMemory.layer(Some(10)),
      
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