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
import zio.langchain.core.memory.*
import zio.langchain.chains.LLMChain
import zio.langchain.memory.BufferMemory
import zio.langchain.integrations.openai.*

import zio.nio.file.{Path, Files}

/**
 * An enhanced Retrieval-Augmented Generation (RAG) example using ZIO LangChain.
 * This example demonstrates how to:
 * 1. Load and process documents with robust error handling
 * 2. Create and use embeddings for semantic search
 * 3. Build a retriever with fallback strategies
 * 4. Stream responses during document retrieval and generation
 * 5. Use memory to maintain context across queries
 * 6. Implement proper error handling and recovery
 */
object EnhancedRAG extends ZIOAppDefault:
  /**
   * The main program.
   */
  override def run: ZIO[Any, Throwable, Unit] =
    // Create the program
    val program = for
      // Print welcome message
      _ <- ZIO.logInfo("Welcome to ZIO LangChain Enhanced RAG Example!")
      _ <- ZIO.logInfo("Loading documents and creating embeddings...")
      
      // Get the services
      llm <- ZIO.service[LLM]
      embeddingModel <- ZIO.service[EmbeddingModel]
      memory <- ZIO.service[Memory]
      
      // Load and process documents with proper error handling
      documentsOrError <- loadDocuments(Path("docs"))
        .catchAll { error =>
          ZIO.logWarning(s"Error loading documents: ${error.getMessage}. Using sample documents instead.") *>
          ZIO.succeed(createSampleDocuments())
        }
      
      _ <- ZIO.logInfo(s"Loaded ${documentsOrError.size} documents")
      
      // Add system message to memory to set context
      _ <- memory.add(ChatMessage(
        role = Role.System,
        content = Some(
          """You are a helpful research assistant. When answering questions, use the context provided.
            |If you don't know the answer based on the provided context, say so rather than making up information.
            |Be clear, concise, and precise in your responses.""".stripMargin)
      ))
      
      // Split documents into chunks with proper error handling
      documentParser = DocumentParsers.byCharacterCount(chunkSize = 1000, chunkOverlap = 200)
      chunksOrError <- documentParser.parseAll(documentsOrError)
        .catchAll { error =>
          ZIO.logWarning(s"Error splitting documents: ${error.getMessage}. Using original documents as chunks.") *>
          ZIO.succeed(documentsOrError.map(doc => 
            Document(doc.id, doc.content, doc.metadata + ("chunk" -> "original"))
          ))
        }
      
      _ <- ZIO.logInfo(s"Created ${chunksOrError.size} chunks from documents")
      
      // Create embeddings for the chunks with progress reporting
      _ <- ZIO.logInfo("Creating embeddings for all chunks... (this may take a moment)")
      embeddedChunksEffect = ZIO.foldLeft(chunksOrError.zipWithIndex)(Vector.empty[(Document, Embedding)]) { 
        case (acc, (chunk, index)) =>
          for
            _ <- if index % 5 == 0 then ZIO.logInfo(s"Processing chunk ${index + 1}/${chunksOrError.size}...") else ZIO.unit
            embedding <- embeddingModel.embedDocument(chunk)
              .catchAll { error =>
                ZIO.logWarning(s"Failed to embed chunk ${index + 1}. Error: ${error.getMessage}") *>
                // Create a fallback embedding (zeros) to prevent the entire process from failing
                ZIO.succeed(Embedding(Vector.fill(1536)(0.0f)))
              }
          yield acc :+ (chunk, embedding)
      }
      
      embeddedChunks <- embeddedChunksEffect
      _ <- ZIO.logInfo("Created embeddings for all chunks")
      
      // Create a retriever with a similarity threshold
      retriever = createEnhancedRetriever(embeddedChunks)
      
      // Create a RAG chain using the LLMChain
      ragChain <- createEnhancedRAGChain(llm, retriever, memory)
      
      // Run the question-answering loop
      _ <- qaLoop(ragChain, llm, memory, retriever)
    yield ()
    
    // Provide the required services and run the program
    program.provide(
      // OpenAI LLM layer
      OpenAILLM.live,
      // OpenAI LLM configuration layer
      ZLayer.succeed(
        OpenAIConfig(
          apiKey = sys.env.getOrElse("OPENAI_API_KEY", ""),
          model = "gpt-4o", // Using a more capable model
          temperature = 0.0 // Low temperature for factual responses
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
      ),
      // Buffer memory layer for maintaining conversation context
      BufferMemory.layer(Some(10)) // Limit to last 10 messages
    )
  
  /**
   * Creates an enhanced retriever with similarity threshold and metadata filtering.
   *
   * @param documents The documents with their embeddings
   * @return A Retriever instance
   */
  private def createEnhancedRetriever(
    documents: Seq[(Document, Embedding)]
  ): Retriever =
    new Retriever:
      override def retrieve(query: String, maxResults: Int): ZIO[Any, RetrieverError, Seq[Document]] =
        for
          // Embed the query
          queryEmbedding <- ZIO.serviceWithZIO[EmbeddingModel](_.embedQuery(query))
            .mapError(e => RetrieverError(e.cause, s"Failed to embed query: ${e.message}"))
            
          // Find most similar documents
          similarities = documents.map { case (doc, embedding) =>
            val score = queryEmbedding.cosineSimilarity(embedding)
            (doc, score)
          }
          
          // Apply a similarity threshold (0.6) and take top results
          filteredDocs = similarities
            .filter { case (_, score) => score > 0.6f }
            .sortBy { case (_, score) => -score } // Sort by descending score
            .take(maxResults)
            .map { case (doc, _) => doc }
            
          // Fallback to top results if no documents match the threshold
          result = if filteredDocs.isEmpty && documents.nonEmpty then
            similarities
              .sortBy { case (_, score) => -score }
              .take(math.min(2, maxResults))
              .map { case (doc, _) => doc }
          else
            filteredDocs
        yield result
  
  /**
   * Creates a more robust RAG Chain using the LLMChain implementation.
   *
   * @param llm The LLM service
   * @param retriever The retriever service
   * @param memory The conversation memory
   * @return A Chain that processes questions and produces answers
   */
  private def createEnhancedRAGChain(
    llm: LLM, 
    retriever: Retriever,
    memory: Memory
  ): ZIO[Any, Nothing, Chain[Any, LangChainError, String, String]] =
    for
      // Chain for memory and retrieval
      queryProcessingChain = Chain[Any, LangChainError, String, (String, Seq[ChatMessage], Seq[Document])] { query =>
        for
          // Get conversation history
          chatHistory <- memory.get.mapError(e => e: LangChainError)
          
          // Retrieve relevant documents
          retrievedDocs <- retriever.retrieve(query, maxResults = 5)
            .mapError(e => e: LangChainError)
            .tap(docs => ZIO.logInfo(s"Retrieved ${docs.size} relevant documents"))
            
        yield (query, chatHistory, retrievedDocs)
      }
      
      // Chain for constructing the prompt with retrieved content
      promptConstructionChain = Chain[Any, LangChainError, (String, Seq[ChatMessage], Seq[Document]), Seq[ChatMessage]] { 
        case (query, chatHistory, docs) =>
          ZIO.succeed {
            // Create context message from retrieved documents
            val contextContent = if docs.isEmpty then
              "No relevant information found in the knowledge base."
            else
              docs.zipWithIndex.map { case (doc, i) =>
                s"""Document ${i + 1}: ${doc.content.trim}
                   |Source: ${doc.metadata.getOrElse("source", "Unknown")}
                   |""".stripMargin
              }.mkString("\n\n")
                
            val contextMessage = ChatMessage(
              role = Role.System,
              content = Some(
                s"""Here is information to help answer the user's question:
                   |
                   |$contextContent
                   |
                   |Use this information to answer the question, but don't reference these documents explicitly in your answer.
                   |If the information doesn't contain the answer, just say so - don't make up information.""".stripMargin
              )
            )
              
            // Add the new question
            val questionMessage = ChatMessage(
              role = Role.User,
              content = Some(query)
            )
              
            // Construct the messages sequence: system + context + chat history + new question
            val systemMessages = chatHistory.filter(_.role == Role.System)
            val nonSystemHistory = chatHistory.filterNot(_.role == Role.System).takeRight(6) // Last 3 exchanges
              
            systemMessages ++ Seq(contextMessage) ++ nonSystemHistory ++ Seq(questionMessage)
          }
      }
      
      // Chain for getting the response from the LLM and storing in memory
      responseChain = Chain[Any, LangChainError, Seq[ChatMessage], String] { messages =>
        // Stream the response for a better user experience
        for
          // Print "Thinking..." before starting generation
          _ <- ZIO.logInfo("\nGenerating response...")
          
          // Build a response buffer
          responseBuffer = new StringBuilder()
          
          // Start streaming the response
          response <- llm.streamCompleteChat(messages)
            .tap { chunk =>
              ZIO.foreach(chunk.message.content) { content =>
                val newContent = content.drop(responseBuffer.length)
                responseBuffer.append(newContent)
                // Print each token as it arrives
                ZIO.logInfo(newContent)
              }.when(chunk.message.content.isDefined)
            }
            .runLast
            .someOrFail(LLMError(new RuntimeException("No response generated")))
            
          // Extract the final message content
          answer = response.message.contentAsString
          
          // Add the assistant's response to memory
          _ <- memory.add(response.message).mapError(e => e: LangChainError)
          
          // Add a newline after the streaming completes
          _ <- ZIO.logInfo("")
        yield answer
      }
    yield
      // Combine the chains
      queryProcessingChain >>> promptConstructionChain >>> responseChain
  
  /**
   * Loads documents from text files in a directory.
   *
   * @param directory The directory containing text files
   * @return A ZIO effect that produces a sequence of documents
   */
  private def loadDocuments(directory: Path): ZIO[Any, DocumentLoaderError, Seq[Document]] =
    for {
      // Check if directory exists
      exists <- Files.exists(directory)
      isDir <- Files.isDirectory(directory).when(exists).map(_.getOrElse(false))
      
      // List files in directory if it exists and is a directory
      docs <- if (exists && isDir) {
        for {
          // List all files in the directory
          paths <- Files.list(directory)
            .map(paths => paths.filter(path =>
              path.toString.endsWith(".txt") || path.toString.endsWith(".md")
            ))
          
          // Read each file and create documents
          docs <- ZIO.foreach(paths.zipWithIndex) { case (path, index) =>
            for {
              content <- Files.readAllLines(path).map(_.mkString("\n"))
              filename <- path.filename
            } yield Document(
              id = s"doc-${index + 1}",
              content = content,
              metadata = Map("source" -> filename.toString, "path" -> path.toString)
            )
          }
        } yield docs
      } else {
        ZIO.succeed(Seq.empty[Document])
      }
    } yield docs
    .mapError(e => DocumentLoaderError(e, s"Failed to load documents from $directory"))
  
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
        metadata = Map("source" -> "zio-overview.txt")
      ),
      Document(
        id = "doc2",
        content = """LangChain is a framework for developing applications powered by language models. 
          |It enables applications that are context-aware, reason, and learn from feedback. 
          |LangChain provides standard interfaces for LLMs, prompt templates, chains, and agents, 
          |making it easy to swap components and reuse code across different applications.""".stripMargin,
        metadata = Map("source" -> "langchain-intro.txt")
      ),
      Document(
        id = "doc3",
        content = """Scala is a high-level programming language that combines object-oriented and 
          |functional programming paradigms. It was designed to be concise, elegant, and type-safe. 
          |Scala runs on the JVM and integrates seamlessly with Java libraries. Its type system supports 
          |both static type-checking and type inference, helping catch errors early while reducing verbosity.""".stripMargin,
        metadata = Map("source" -> "scala-intro.txt")
      ),
      Document(
        id = "doc4",
        content = """Retrieval-Augmented Generation (RAG) is a technique that enhances Large Language Models 
          |by retrieving relevant information from external knowledge sources before generating responses. 
          |This approach helps ground the model's outputs in factual, up-to-date information, reducing 
          |hallucinations and improving accuracy. In a typical RAG system, user queries are used to search 
          |and retrieve relevant documents, which are then provided as context to the LLM.""".stripMargin,
        metadata = Map("source" -> "rag-explanation.txt")
      ),
      Document(
        id = "doc5",
        content = """Embeddings are numerical representations of text that capture semantic meaning. 
          |They convert words, phrases, or documents into dense vectors in a high-dimensional space, 
          |where semantically similar text has similar vector representations. This enables powerful 
          |applications like semantic search, where documents are retrieved based on meaning rather than 
          |just keyword matching. Modern embedding models like those from OpenAI can capture nuanced 
          |relationships between different pieces of text.""".stripMargin,
        metadata = Map("source" -> "embeddings-explained.txt")
      )
    )
  
  /**
   * The question-answering loop with enhanced error handling.
   * It repeatedly prompts the user for questions and displays the answers.
   *
   * @param ragChain The RAG chain
   * @param llm The LLM service for fallback responses on errors
   * @param memory The memory service
   * @param retriever The retriever for diagnostics on errors
   * @return A ZIO effect that completes when the user exits
   */
  private def qaLoop(
    ragChain: Chain[Any, LangChainError, String, String],
    llm: LLM,
    memory: Memory,
    retriever: Retriever
  ): ZIO[Any, Throwable, Unit] =
    for
      // Prompt the user for a question
      _ <- ZIO.logInfo("\nEnter your question (or 'exit' to quit, 'clear' to reset conversation):")
      _ <- ZIO.logInfo("> ")
      input <- readLine
      
      // Process the input
      result <- input.toLowerCase match
        case "exit" =>
          ZIO.logInfo("Goodbye!")
        
        case "clear" =>
          // Reset the conversation
          memory.clear *>
          ZIO.logInfo("Conversation history cleared.") *>
          qaLoop(ragChain, llm, memory, retriever)
          
        case question =>
          for
            // Add the user question to memory
            _ <- memory.add(ChatMessage.user(question))
            
            // Process the question with error handling
            response <- ragChain.run(question).catchAll { error =>
              ZIO.logError(s"\nError: ${error.message}") *>
              // Provide a graceful fallback
              memory.get.flatMap { messages =>
                llm.completeChat(
                  messages :+ ChatMessage(
                    role = Role.System,
                    content = Some("There was an issue processing the request. Please provide a helpful response anyway.")
                  )
                )
              }.map(_.message.contentAsString)
            }
            
            // Continue the loop
            result <- qaLoop(ragChain, llm, memory, retriever)
          yield result
    yield ()