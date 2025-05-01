package zio.langchain.examples

import zio.*
import zio.Console.*
import zio.http.Client

import zio.langchain.core.model.*
import zio.langchain.core.document.*
import zio.langchain.core.domain.*
import zio.langchain.integrations.openai.*

/**
 * Example demonstrating a simple Retrieval-Augmented Generation (RAG) system.
 *
 * This example shows how to:
 * 1. Create document embeddings
 * 2. Implement similarity search for document retrieval
 * 3. Use retrieved context to enhance LLM responses
 */
object SimpleRAG extends ZIOAppDefault {

  // Sample documents
  val documents = Seq(
    Document(
      id = "doc1",
      content = "ZIO is a library for asynchronous and concurrent programming in Scala.",
      metadata = Map("source" -> "docs")
    ),
    Document(
      id = "doc2",
      content = "ZIO provides a powerful effect system for functional programming.",
      metadata = Map("source" -> "docs")
    ),
    Document(
      id = "doc3",
      content = "ZIO LangChain is a Scala wrapper around langchain4j with ZIO integration.",
      metadata = Map("source" -> "docs")
    )
  )

  /**
   * Retrieve relevant documents based on similarity to the query
   */
  def retrieveSimilarDocuments(
    query: String, 
    embeddedDocs: Seq[(Document, Embedding)],
    embeddingModel: EmbeddingModel,
    maxResults: Int = 2
  ): ZIO[Any, Throwable, Seq[Document]] = {
    for {
      // Create embedding for the query
      queryEmbedding <- embeddingModel.embed(query)
      
      // Calculate similarities
      similarities = embeddedDocs.map { case (doc, embedding) =>
        (doc, queryEmbedding.cosineSimilarity(embedding))
      }
      
      // Sort by similarity (highest first) and take top results
      topDocs = similarities.sortBy(-_._2).take(maxResults).map(_._1)
    } yield topDocs
  }

  /**
   * Ask a question using retrieved context
   */
  def askWithContext(
    question: String, 
    context: Seq[Document], 
    llm: LLM
  ): ZIO[Any, Throwable, String] = {
    // Format context for the prompt
    val contextText = context.map(_.content).mkString("\n\n")
    
    // Create prompt with context
    val prompt = s"""Based on the following information:
                   |
                   |$contextText
                   |
                   |Question: $question
                   |
                   |Answer:""".stripMargin
    
    // Get response from LLM
    llm.complete(prompt)
  }

  override def run = {
    // Define our program
    val program = for {
      // Print welcome message
      _ <- printLine("Welcome to ZIO LangChain SimpleRAG Example!")
      
      // Get the models
      embeddingModel <- ZIO.service[EmbeddingModel]
      llm <- ZIO.service[LLM]
      
      // Create embeddings for documents
      _ <- printLine("Creating embeddings for sample documents...")
      embeddedDocs <- embeddingModel.embedDocuments(documents)
      _ <- printLine("Embeddings created successfully.")
      
      // Example question
      question = "What is ZIO?"
      _ <- printLine(s"Asking question: '$question'")
      
      // Retrieve relevant documents
      relevantDocs <- retrieveSimilarDocuments(question, embeddedDocs, embeddingModel)
      _ <- printLine(s"Retrieved ${relevantDocs.size} relevant documents")
      
      // Get answer using RAG
      answer <- askWithContext(question, relevantDocs, llm)
      
      // Display answer
      _ <- printLine("Answer:")
      _ <- printLine(answer)
      
      // Final message
      _ <- printLine("\nSimpleRAG example completed successfully.")
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