package zio.langchain.core

import scala.collection.immutable.Map

/**
 * Core domain types for ZIO LangChain.
 */
object domain:
  /**
   * Represents the role of a message in a chat conversation.
   */
  enum Role:
    case User, Assistant, System, Tool, Function

  /**
   * Represents a function parameter.
   *
   * @param name The name of the parameter
   * @param description The description of the parameter
   * @param required Whether the parameter is required
   * @param `type` The type of the parameter (e.g., "string", "number", "boolean")
   * @param enum Possible values if this is an enum parameter
   */
  case class FunctionParameter(
    name: String,
    description: String,
    required: Boolean = false,
    `type`: String = "string",
    possibleValues: Option[Seq[String]] = None
  )

  /**
   * Represents a function definition that can be used by an LLM.
   *
   * @param name The name of the function
   * @param description The description of what the function does
   * @param parameters The parameters of the function
   */
  case class FunctionDefinition(
    name: String,
    description: String,
    parameters: Seq[FunctionParameter]
  )

  /**
   * Represents a function call within a chat message.
   *
   * @param name The name of the function to call
   * @param arguments The arguments to pass to the function as a JSON string
   */
  case class FunctionCall(
    name: String,
    arguments: String
  )

  /**
   * Represents a tool that can be called by an LLM.
   *
   * @param type The type of tool
   * @param function The function definition for this tool
   */
  case class ToolDefinition(
    `type`: String = "function",
    function: FunctionDefinition
  )

  /**
   * Represents a message in a chat conversation.
   *
   * @param role The role of the message sender
   * @param content The content of the message
   * @param metadata Additional metadata associated with the message
   * @param functionCall Optional function call if this is a function call message
   * @param name Optional name, used for function messages to specify which function is returning a result
   * @param toolCalls Optional tool calls if this is a tool call message
   */
  case class ChatMessage(
    role: Role,
    content: Option[String] = None,
    metadata: Map[String, String] = Map.empty,
    functionCall: Option[FunctionCall] = None,
    name: Option[String] = None,
    toolCalls: Option[Seq[ToolCall]] = None
  ):
    def withContent(content: String): ChatMessage = this.copy(content = Some(content))
    
    def contentAsString: String = content.getOrElse("")

  /**
   * Represents a tool call within a message.
   *
   * @param id The ID of the tool call
   * @param type The type of tool call (usually "function")
   * @param function The function being called
   */
  case class ToolCall(
    id: String,
    `type`: String = "function",
    function: FunctionCall
  )

  object ChatMessage:
    def user(content: String): ChatMessage = ChatMessage(Role.User, Some(content))
    
    def assistant(content: String): ChatMessage = ChatMessage(Role.Assistant, Some(content))
    
    def system(content: String): ChatMessage = ChatMessage(Role.System, Some(content))
    
    def tool(content: String): ChatMessage = ChatMessage(Role.Tool, Some(content))
    
    def function(name: String, content: String): ChatMessage =
      ChatMessage(Role.Function, Some(content), name = Some(name))
    
    def functionCall(name: String, arguments: String): ChatMessage =
      ChatMessage(Role.Assistant, functionCall = Some(FunctionCall(name, arguments)))
    
    def toolCall(id: String, name: String, arguments: String): ChatMessage =
      ChatMessage(
        Role.Assistant,
        toolCalls = Some(Seq(ToolCall(id, "function", FunctionCall(name, arguments))))
      )

  /**
   * Represents a response from an AI chat model.
   *
   * @param message The message from the AI
   * @param usage Token usage information
   * @param finishReason Optional reason why the generation finished
   */
  case class ChatResponse(
    message: ChatMessage,
    usage: TokenUsage,
    finishReason: Option[String] = None
  )
  
  /**
   * Represents token usage information for a model response.
   *
   * @param promptTokens Number of tokens in the prompt
   * @param completionTokens Number of tokens in the completion
   * @param totalTokens Total number of tokens used
   */
  case class TokenUsage(
    promptTokens: Int,
    completionTokens: Int,
    totalTokens: Int
  )
  
  /**
   * Represents a document with content and metadata.
   *
   * @param id Unique identifier for the document
   * @param content The text content of the document
   * @param metadata Additional metadata associated with the document
   */
  case class Document(
    id: String,
    content: String,
    metadata: Map[String, String] = Map.empty
  )
  
  /**
   * Represents a vector embedding.
   * This is an opaque type to ensure type safety and encapsulation.
   */
  opaque type Embedding = Vector[Float]
  
  object Embedding:
    /**
     * Creates a new embedding from a vector of float values.
     *
     * @param values The vector values
     * @return A new Embedding
     */
    def apply(values: Vector[Float]): Embedding = values
    
    /**
     * Extension methods for Embedding.
     */
    extension (e: Embedding)
      /**
       * Gets the underlying vector values.
       *
       * @return The vector values
       */
      def values: Vector[Float] = e
      
      /**
       * Gets the dimension of the embedding.
       *
       * @return The dimension
       */
      def dimension: Int = e.size
      
      /**
       * Calculates the cosine similarity between this embedding and another.
       *
       * @param other The other embedding
       * @return The cosine similarity value between -1 and 1
       */
      def cosineSimilarity(other: Embedding): Float =
        val dotProduct = (e zip other).map { case (a, b) => a * b }.sum
        val magnitudeA = math.sqrt(e.map(x => x * x).sum).toFloat
        val magnitudeB = math.sqrt(other.map(x => x * x).sum).toFloat
        
        if (magnitudeA == 0 || magnitudeB == 0) 0f
        else dotProduct / (magnitudeA * magnitudeB)
  
  /**
   * Model parameters for LLM requests.
   * This is a base trait that can be extended by model-specific parameter classes.
   */
  trait ModelParameters:
    /**
     * Converts the parameters to a map that can be passed to the underlying model implementation.
     */
    def toMap: Map[String, Any]

  /**
   * Default implementation of ModelParameters.
   *
   * @param temperature Controls randomness: 0 is deterministic, higher values increase randomness
   * @param topP Controls diversity via nucleus sampling: 0.1 means only considering the tokens with top 10% probability
   * @param maxTokens Maximum number of tokens to generate
   * @param presencePenalty Penalizes repeated tokens: higher values increase penalization
   * @param frequencyPenalty Penalizes frequent tokens: higher values increase penalization
   * @param seed Random seed for deterministic generation
   */
  case class DefaultModelParameters(
    temperature: Option[Double] = None,
    topP: Option[Double] = None,
    maxTokens: Option[Int] = None,
    presencePenalty: Option[Double] = None,
    frequencyPenalty: Option[Double] = None,
    seed: Option[Long] = None
  ) extends ModelParameters:
    override def toMap: Map[String, Any] = Map.newBuilder[String, Any]
      .addIfDefined("temperature", temperature)
      .addIfDefined("top_p", topP)
      .addIfDefined("max_tokens", maxTokens)
      .addIfDefined("presence_penalty", presencePenalty)
      .addIfDefined("frequency_penalty", frequencyPenalty)
      .addIfDefined("seed", seed)
      .result()

  extension[K, V] (builder: scala.collection.mutable.Builder[(K, V), Map[K, V]])
    def addIfDefined(key: K, value: Option[V]): builder.type =
      value.foreach(v => builder += ((key, v)))
      builder