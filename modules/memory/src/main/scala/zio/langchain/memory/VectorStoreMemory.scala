package zio.langchain.memory

import zio.*
import zio.langchain.core.memory.Memory
import zio.langchain.core.domain.*
import zio.langchain.core.errors.*
import zio.langchain.core.model.EmbeddingModel
import zio.langchain.core.retriever.Retriever

import java.util.UUID
import java.util.concurrent.TimeUnit

/**
 * A memory implementation that stores embeddings of messages for semantic lookup.
 * This allows for more intelligent retrieval of past conversation context based on relevance.
 *
 * @param messages The reference to the current messages (for maintaining order)
 * @param embeddingModel The embedding model to use for generating embeddings
 * @param vectorStore The vector store to use for storing and retrieving embeddings
 * @param relevanceThreshold The minimum similarity score for a message to be considered relevant
 * @param maxRetrievedMessages The maximum number of messages to retrieve from the vector store
 */
class VectorStoreMemory private (
  messages: Ref[Vector[ChatMessage]],
  embeddingModel: EmbeddingModel,
  vectorStore: Retriever,
  relevanceThreshold: Double = 0.7,
  maxRetrievedMessages: Int = 5
) extends Memory:
  /**
   * Adds a message to the conversation history and stores its embedding in the vector store.
   *
   * @param message The chat message to add
   * @return A ZIO effect that completes with unit or fails with a MemoryError
   */
  override def add(message: ChatMessage): ZIO[Any, MemoryError, Unit] =
    for {
      // Add message to the buffer
      _ <- messages.update(_ :+ message)
      
      // Create a document from the message
      messageContent = message.contentAsString
      messageId = UUID.randomUUID().toString
      document = Document(
        id = messageId,
        content = messageContent,
        metadata = Map(
          "role" -> message.role.toString,
          "timestamp" -> java.lang.System.currentTimeMillis().toString
        ) ++ message.metadata
      )
      
      // Store the document in the vector store
      // We need to handle this differently since not all Retrievers support addDocument
      // This is a temporary solution until we have a proper VectorStore trait
      _ <- ZIO.attempt {
        // Use reflection to check if the vectorStore has an addDocument method
        val method = vectorStore.getClass.getMethod("addDocument", classOf[Document])
        method.invoke(vectorStore, document).asInstanceOf[ZIO[Any, RetrieverError, Unit]]
      }.flatten
        .mapError(err => MemoryError(err, "Failed to store message in vector store"))
        .catchAll(err =>
          ZIO.logWarning(s"Failed to add document to vector store: ${err.getMessage}") *>
          ZIO.fail(err)
        )
    } yield ()
  
  /**
   * Retrieves messages from the conversation history, including semantically relevant messages.
   *
   * @return A ZIO effect that produces a sequence of chat messages or fails with a MemoryError
   */
  override def get: ZIO[Any, MemoryError, Seq[ChatMessage]] =
    messages.get
  
  /**
   * Retrieves messages relevant to a query from the conversation history.
   *
   * @param query The query to find relevant messages for
   * @return A ZIO effect that produces a sequence of chat messages or fails with a MemoryError
   */
  def getRelevantMessages(query: String): ZIO[Any, MemoryError, Seq[ChatMessage]] =
    for {
      // Retrieve relevant documents from the vector store
      relevantDocsWithScores <- vectorStore.retrieveWithScores(query, maxRetrievedMessages)
        .mapError(err => MemoryError(err, "Failed to retrieve relevant messages from vector store"))
      
      // Filter by relevance threshold
      relevantDocs = relevantDocsWithScores
        .filter { case (_, score) => score >= relevanceThreshold }
        .map { case (doc, _) => doc }
      
      // Convert documents back to chat messages
      relevantMessages = relevantDocs.map { doc =>
        val role = doc.metadata.get("role")
          .flatMap(r => Role.values.find(_.toString == r))
          .getOrElse(Role.User)
        
        ChatMessage(
          role = role,
          content = Some(doc.content),
          metadata = doc.metadata.filter { case (k, _) => k != "role" && k != "timestamp" }
        )
      }
    } yield relevantMessages
  
  /**
   * Clears the conversation history and removes all messages from the vector store.
   *
   * @return A ZIO effect that completes with unit or fails with a MemoryError
   */
  override def clear: ZIO[Any, MemoryError, Unit] =
    for {
      // Clear the message buffer
      _ <- messages.set(Vector.empty)
      
      // Clear the vector store (if supported)
      // We need to handle this differently since not all Retrievers support deleteAll
      _ <- ZIO.attempt {
        // Use reflection to check if the vectorStore has a deleteAll method
        val method = vectorStore.getClass.getMethod("deleteAll")
        method.invoke(vectorStore).asInstanceOf[ZIO[Any, RetrieverError, Unit]]
      }.flatten
        .mapError(err => MemoryError(err, "Failed to clear vector store"))
        .catchAll(err => ZIO.logWarning(s"Failed to clear vector store: ${err.getMessage}").as(()))
    } yield ()

/**
 * Companion object for VectorStoreMemory.
 */
object VectorStoreMemory:
  /**
   * Creates a new VectorStoreMemory instance.
   *
   * @param embeddingModel The embedding model to use
   * @param vectorStore The vector store to use
   * @param relevanceThreshold The minimum similarity score for a message to be considered relevant
   * @param maxRetrievedMessages The maximum number of messages to retrieve from the vector store
   * @return A ZIO effect that produces a VectorStoreMemory
   */
  def make(
    embeddingModel: EmbeddingModel,
    vectorStore: Retriever,
    relevanceThreshold: Double = 0.7,
    maxRetrievedMessages: Int = 5
  ): UIO[VectorStoreMemory] =
    for {
      messagesRef <- Ref.make(Vector.empty[ChatMessage])
    } yield new VectorStoreMemory(
      messagesRef,
      embeddingModel,
      vectorStore,
      relevanceThreshold,
      maxRetrievedMessages
    )
  
  /**
   * Creates a new VectorStoreMemory instance with initial messages.
   *
   * @param initialMessages The initial messages to store
   * @param embeddingModel The embedding model to use
   * @param vectorStore The vector store to use
   * @param relevanceThreshold The minimum similarity score for a message to be considered relevant
   * @param maxRetrievedMessages The maximum number of messages to retrieve from the vector store
   * @return A ZIO effect that produces a VectorStoreMemory
   */
  def makeWithInitialMessages(
    initialMessages: Seq[ChatMessage],
    embeddingModel: EmbeddingModel,
    vectorStore: Retriever,
    relevanceThreshold: Double = 0.7,
    maxRetrievedMessages: Int = 5
  ): ZIO[Any, MemoryError, VectorStoreMemory] =
    for {
      // Create the memory instance
      messagesRef <- Ref.make(Vector.empty[ChatMessage])
      memory = new VectorStoreMemory(
        messagesRef,
        embeddingModel,
        vectorStore,
        relevanceThreshold,
        maxRetrievedMessages
      )
      
      // Add initial messages
      _ <- ZIO.foreachDiscard(initialMessages)(memory.add)
    } yield memory
  
  /**
   * Creates a ZLayer that provides a Memory implementation using VectorStoreMemory.
   *
   * @param embeddingModel The embedding model to use
   * @param vectorStore The vector store to use
   * @param relevanceThreshold The minimum similarity score for a message to be considered relevant
   * @param maxRetrievedMessages The maximum number of messages to retrieve from the vector store
   * @return A ZLayer that provides a Memory
   */
  def layer(
    embeddingModel: EmbeddingModel,
    vectorStore: Retriever,
    relevanceThreshold: Double = 0.7,
    maxRetrievedMessages: Int = 5
  ): ULayer[Memory] =
    ZLayer.fromZIO(make(embeddingModel, vectorStore, relevanceThreshold, maxRetrievedMessages))
  
  /**
   * Creates a ZLayer that provides a Memory implementation using VectorStoreMemory with initial messages.
   *
   * @param initialMessages The initial messages to store
   * @param embeddingModel The embedding model to use
   * @param vectorStore The vector store to use
   * @param relevanceThreshold The minimum similarity score for a message to be considered relevant
   * @param maxRetrievedMessages The maximum number of messages to retrieve from the vector store
   * @return A ZLayer that provides a Memory
   */
  def layerWithInitialMessages(
    initialMessages: Seq[ChatMessage],
    embeddingModel: EmbeddingModel,
    vectorStore: Retriever,
    relevanceThreshold: Double = 0.7,
    maxRetrievedMessages: Int = 5
  ): ZLayer[Any, MemoryError, Memory] =
    ZLayer.fromZIO(makeWithInitialMessages(
      initialMessages, 
      embeddingModel, 
      vectorStore, 
      relevanceThreshold, 
      maxRetrievedMessages
    ))
  
  /**
   * Creates a ZLayer that provides a Memory implementation using VectorStoreMemory.
   * This version takes an EmbeddingModel and Retriever from the environment.
   *
   * @param relevanceThreshold The minimum similarity score for a message to be considered relevant
   * @param maxRetrievedMessages The maximum number of messages to retrieve from the vector store
   * @return A ZLayer that requires an EmbeddingModel and Retriever and provides a Memory
   */
  def layerFromServices(
    relevanceThreshold: Double = 0.7,
    maxRetrievedMessages: Int = 5
  ): ZLayer[EmbeddingModel & Retriever, Nothing, Memory] =
    ZLayer.fromZIO(
      for {
        embeddingModel <- ZIO.service[EmbeddingModel]
        vectorStore <- ZIO.service[Retriever]
        memory <- make(embeddingModel, vectorStore, relevanceThreshold, maxRetrievedMessages)
      } yield memory
    )