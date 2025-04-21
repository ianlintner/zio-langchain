package zio.langchain.examples

import zio.*
import zio.Console.*

import zio.langchain.core.model.*
import zio.langchain.core.retriever.*
import zio.langchain.core.document.*
import zio.langchain.core.domain.*
import zio.langchain.core.chain.*
import zio.langchain.integrations.openai.*
import zio.langchain.integrations.pinecone.*
import zio.langchain.core.errors.{LangChainError, RetrieverError, LLMError}

/**
 * An example demonstrating the use of Pinecone as a vector store for RAG.
 * This example shows how to:
 * 1. Configure and initialize the Pinecone vector store
 * 2. Add documents to the store
 * 3. Retrieve documents based on semantic similarity
 * 4. Use the Pinecone store in a RAG application
 */
object PineconeExample extends ZIOAppDefault:
  /**
   * The main program.
   */
  override def run: ZIO[Any, Throwable, Unit] =
    // Create the program
    val program = for
      // Print welcome message
      _ <- ZIO.logInfo("Welcome to ZIO LangChain Pinecone Example!")
      
      // Get the services
      llm <- ZIO.service[LLM]
      embeddingModel <- ZIO.service[EmbeddingModel]
      pineconeStore <- ZIO.service[PineconeStore]
      
      // Create sample documents
      documents = createSampleDocuments()
      _ <- ZIO.logInfo(s"Created ${documents.size} sample documents")
      
      // Add documents to Pinecone
      _ <- ZIO.logInfo("Adding documents to Pinecone...")
      _ <- pineconeStore.addDocuments(documents)
        .catchAll { error =>
          ZIO.logError(s"Error adding documents to Pinecone: ${error.getMessage}") *>
          ZIO.fail(error)
        }
      _ <- ZIO.logInfo("Documents added successfully")
      
      // Create a RAG chain using the Pinecone store as a retriever
      ragChain = createRAGChain(llm, pineconeStore)
      
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
          model = "gpt-4o",
          temperature = 0.0 // Use a low temperature for factual responses
        )
      ),
      // OpenAI Embedding layer
      OpenAIEmbedding.live,
      // OpenAI Embedding configuration layer
      ZLayer.succeed(
        OpenAIEmbeddingConfig(
          apiKey = sys.env.getOrElse("OPENAI_API_KEY", ""),
          model = "text-embedding-3-large"
        )
      ),
      // Pinecone store layer
      PineconeStore.liveStore,
      // Pinecone configuration layer
      ZLayer.succeed(
        PineconeConfig(
          apiKey = sys.env.getOrElse("PINECONE_API_KEY", ""),
          environment = sys.env.getOrElse("PINECONE_ENVIRONMENT", ""),
          projectId = sys.env.getOrElse("PINECONE_PROJECT_ID", ""),
          indexName = sys.env.getOrElse("PINECONE_INDEX_NAME", ""),
          namespace = sys.env.get("PINECONE_NAMESPACE")
        )
      )
    )
  
  /**
   * Creates sample documents for demonstration.
   *
   * @return A sequence of documents
   */
  private def createSampleDocuments(): Seq[Document] =
    Seq(
      Document(
        id = "doc1",
        content = """ZIO is a library for asynchronous and concurrent programming in Scala.
                    |ZIO provides a simple, type-safe, testable, and performant foundation for
                    |building asynchronous and concurrent applications. ZIO represents effects
                    |as values, making them easy to compose, test, and reason about.""".stripMargin,
        metadata = Map("source" -> "zio-docs", "topic" -> "programming")
      ),
      Document(
        id = "doc2",
        content = """Pinecone is a vector database designed for machine learning applications.
                    |It provides fast and scalable vector similarity search, which is essential
                    |for applications like semantic search, recommendation systems, and
                    |retrieval-augmented generation (RAG).""".stripMargin,
        metadata = Map("source" -> "pinecone-docs", "topic" -> "database")
      ),
      Document(
        id = "doc3",
        content = """Scala is a programming language that combines object-oriented and functional
                    |programming paradigms. It is designed to be concise, elegant, and type-safe.
                    |Scala runs on the JVM and can interoperate with Java libraries.""".stripMargin,
        metadata = Map("source" -> "scala-docs", "topic" -> "programming")
      ),
      Document(
        id = "doc4",
        content = """Retrieval-Augmented Generation (RAG) is a technique that enhances large
                    |language models by retrieving relevant information from external knowledge
                    |sources. This approach improves the factuality and relevance of generated
                    |responses by grounding them in accurate, up-to-date information.""".stripMargin,
        metadata = Map("source" -> "ai-research", "topic" -> "ai")
      ),
      Document(
        id = "doc5",
        content = """Vector embeddings are numerical representations of data (like text, images,
                    |or audio) in a high-dimensional space. These embeddings capture semantic
                    |meaning, allowing for operations like similarity search. In the context of
                    |natural language processing, similar concepts have similar vector
                    |representations.""".stripMargin,
        metadata = Map("source" -> "ml-docs", "topic" -> "ai")
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
      val docsText = documents.map { doc =>
        val source = doc.metadata.getOrElse("source", "Unknown")
        val topic = doc.metadata.getOrElse("topic", "Unknown")
        s"""Content: ${doc.content}
           |Source: $source
           |Topic: $topic""".stripMargin
      }.mkString("\n\n")
      
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
            .catchAll { error =>
              ZIO.logError(s"Error: ${error.getMessage}") *>
              ZIO.succeed("I encountered an error while trying to answer your question. Please try again.")
            }
          
          // Display the answer
          _ <- ZIO.logInfo("\nAnswer:")
          _ <- ZIO.logInfo(answer)
          
          // Continue the loop
          result <- qaLoop(ragChain)
        yield result
    yield result