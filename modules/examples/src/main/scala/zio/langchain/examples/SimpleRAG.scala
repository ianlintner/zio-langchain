package zio.langchain.examples

import zio.*
import zio.Console.*

import zio.langchain.core.model.*
import zio.langchain.core.retriever.*
import zio.langchain.core.document.*
import zio.langchain.core.domain.*
import zio.langchain.core.chain.*
import zio.langchain.integrations.openai.*
import zio.langchain.core.errors.{LangChainError, RetrieverError, LLMError}

import java.nio.file.{Path, Files}

/**
 * A simple Retrieval-Augmented Generation (RAG) example using ZIO LangChain.
 * This example demonstrates how to:
 * 1. Load documents from text files
 * 2. Split documents into chunks
 * 3. Create embeddings for the chunks
 * 4. Build a retriever that finds relevant documents
 * 5. Create a RAG chain that uses the retriever and an LLM to answer questions
 */
object SimpleRAG extends ZIOAppDefault:
  /**
   * The main program.
   */
  override def run: ZIO[Any, Throwable, Unit] =
    // Create the program
    val program = for
      // Print welcome message
      _ <- ZIO.logInfo("Welcome to ZIO LangChain RAG Example!")
      _ <- ZIO.logInfo("Loading documents and creating embeddings...")
      
      // Get the services
      llm <- ZIO.service[LLM]
      embeddingModel <- ZIO.service[EmbeddingModel]
      
      // Load and process documents
      documents <- loadDocuments(Path.of("docs"))
      _ <- ZIO.logInfo(s"Loaded ${documents.size} documents")
      
      // Split documents into chunks
      documentParser = DocumentParsers.byCharacterCount(chunkSize = 1000, chunkOverlap = 200)
      chunks <- documentParser.parseAll(documents)
      _ <- ZIO.logInfo(s"Split into ${chunks.size} chunks")
      
      // Create embeddings for the chunks
      embeddedChunks <- embeddingModel.embedDocuments(chunks)
      _ <- ZIO.logInfo("Created embeddings for all chunks")
      // Create an in-memory retriever
      retriever = new InMemoryRetriever(embeddedChunks)
      
      // Create a RAG chain
      ragChain = createRAGChain(llm, retriever)
      
      // Run the question-answering loop
      _ <- qaLoop(ragChain)
    yield ()
    
    // Provide the required services and run the program
    program.provide(
      // OpenAI LLM layer
      OpenAILLM.live,
      // OpenAI LLM configuration layer
      ZLayer.succeed(
        OpenAIConfig(
          apiKey = sys.env.getOrElse("OPENAI_API_KEY", ""),
          model = "gpt-3.5-turbo",
          temperature = 0.0 // Use a low temperature for factual responses
        )
      ),
      // OpenAI Embedding layer
      OpenAIEmbedding.live,
      // OpenAI Embedding configuration layer
      ZLayer.succeed(
        OpenAIEmbeddingConfig(
          apiKey = sys.env.getOrElse("OPENAI_API_KEY", ""),
          model = "text-embedding-ada-002"
        )
      )
    )
  
  /**
   * Loads documents from text files in a directory.
   *
   * @param directory The directory containing text files
   * @return A ZIO effect that produces a sequence of documents
   */
  private def loadDocuments(directory: Path): ZIO[Any, LangChainError, Seq[Document]] =
    // This is a simplified implementation
    // In a real application, you would use DocumentLoader to load files from the directory
    ZIO.succeed(
      Seq(
        Document(
          id = "doc1",
          content = "ZIO is a library for asynchronous and concurrent programming in Scala. " +
                   "ZIO provides a simple, type-safe, testable, and performant foundation for " +
                   "building asynchronous and concurrent applications.",
          metadata = Map("source" -> "zio-intro.txt")
        ),
        Document(
          id = "doc2",
          content = "LangChain is a framework for developing applications powered by language models. " +
                   "It enables applications that are context-aware, reason, and learn from feedback.",
          metadata = Map("source" -> "langchain-intro.txt")
        ),
        Document(
          id = "doc3",
          content = "Scala is a programming language that combines object-oriented and functional programming. " +
                   "It is designed to be concise, elegant, and type-safe.",
          metadata = Map("source" -> "scala-intro.txt")
        )
      )
    )
  
  /**
   * Creates a RAG chain that uses a retriever and an LLM to answer questions.
   *
   * @param llm The LLM service
   * @param retriever The retriever service
   * @return A Chain that takes a question and produces an answer
   */
  private def createRAGChain(llm: LLM, retriever: Retriever): Chain[Any, LangChainError, String, String] =
    // Create a chain that retrieves relevant documents
    val retrievalChain = Chain[Any, LangChainError, String, Seq[Document]] { query =>
      retriever.retrieve(query, maxResults = 3)
        .mapError(e => e: LangChainError)
    }
    
    // Create a chain that formats the prompt with the retrieved documents
    val promptChain = Chain[Any, LangChainError, Seq[Document], String] { documents =>
      val docsText = documents.map(doc => s"Content: ${doc.content}").mkString("\n\n")
      ZIO.succeed(
        s"""Answer the question based only on the following context:
           |
           |$docsText
           |
           |Question: {{question}}
           |
           |Answer:""".stripMargin
      )
    }
    
    // Create a chain that replaces the {{question}} placeholder with the actual question
    val templateChain = Chain[Any, LangChainError, (String, String), String] { case (template, question) =>
      ZIO.succeed(template.replace("{{question}}", question))
    }
    
    // Create a chain that sends the prompt to the LLM
    val llmChain = Chain[Any, LangChainError, String, String] { prompt =>
      llm.complete(prompt)
        .mapError(e => e: LangChainError)
    }
    
    // Combine the chains
    Chain[Any, LangChainError, String, (String, Seq[Document])] { query =>
      retrievalChain.run(query).map(docs => (query, docs))
    } >>> Chain[Any, LangChainError, (String, Seq[Document]), (String, String)] { case (query, docs) =>
      promptChain.run(docs).map(template => (template, query))
    } >>> templateChain >>> llmChain
  
  /**
   * A simple in-memory retriever that finds relevant documents based on embedding similarity.
   *
   * @param documents The documents with their embeddings
   */
  private class InMemoryRetriever(documents: Seq[(Document, Embedding)]) extends Retriever:
    override def retrieve(query: String, maxResults: Int): ZIO[Any, zio.langchain.core.errors.RetrieverError, Seq[Document]] =
      // In a real application, you would use the embedding model to embed the query
      // and find the most similar documents
      ZIO.succeed(documents.map(_._1).take(maxResults))
  
  /**
   * The question-answering loop.
   * It repeatedly prompts the user for questions and displays the answers.
   *
   * @param ragChain The RAG chain
   * @return A ZIO effect that completes when the user exits
   */
  private def qaLoop(ragChain: Chain[Any, LangChainError, String, String]): ZIO[Any, Throwable, Unit] =
    for
      // Prompt the user for a question
      _ <- ZIO.logInfo("\nEnter your question (or 'exit' to quit):")
      _ <- ZIO.logInfo("> ")
      question <- readLine
      
      // Check if the user wants to exit
      result <- if question.toLowerCase == "exit" then
        ZIO.logInfo("Goodbye!").as(())
      else
        for
          // Process the question and get the answer
          _ <- ZIO.logInfo("Thinking...")
          answer <- ragChain.run(question)
          
          // Display the answer
          _ <- ZIO.logInfo("\nAnswer:")
          _ <- ZIO.logInfo(answer)
          
          // Continue the loop
          result <- qaLoop(ragChain)
        yield result
    yield result