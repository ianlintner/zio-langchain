package zio.langchain.examples

import zio.*
import zio.json.*
import zio.Console.*

import zio.langchain.core.model.LLM
import zio.langchain.core.domain.ChatMessage
import zio.langchain.integrations.openai.{OpenAIConfig, OpenAILLM}

/**
 * Example demonstrating how to work with structured outputs in ZIO LangChain.
 * This example shows how to:
 * 1. Define a case class for structured data
 * 2. Create a prompt that asks for structured data
 * 3. Parse the response into the structured data type
 */
object StructuredOutputExample extends ZIOAppDefault:
  /**
   * Case class representing a movie review.
   */
  case class MovieReview(
    title: String,
    year: Int,
    rating: Double,
    review: String
  )

  /**
   * JSON encoder/decoder for MovieReview.
   */
  given JsonEncoder[MovieReview] = DeriveJsonEncoder.gen[MovieReview]
  given JsonDecoder[MovieReview] = DeriveJsonDecoder.gen[MovieReview]

  /**
   * Simple example of getting a structured response from the LLM.
   */
  def simpleExample(llm: LLM): ZIO[Any, Throwable, Unit] = {
    // Define the prompt
    val prompt = 
      "Generate a movie review in JSON format. The review should include:\n" +
      "1. title: The title of the movie\n" +
      "2. year: The year the movie was released (as a number)\n" +
      "3. rating: A rating from 0.0 to 10.0 (as a number)\n" +
      "4. review: A short review of the movie\n\n" +
      "Format the response as valid JSON that matches this structure:\n" +
      "{\n" +
      "  \"title\": \"Movie Title\",\n" +
      "  \"year\": 2023,\n" +
      "  \"rating\": 8.5,\n" +
      "  \"review\": \"This is a review of the movie.\"\n" +
      "}"
    
    for {
      // Get the response from the LLM
      response <- llm.complete(prompt)
      _ <- printLine(s"LLM Response:\n$response\n")
      
      // Extract the JSON part from the response
      jsonStr = extractJson(response)
      _ <- printLine(s"Extracted JSON:\n$jsonStr\n")
      
      // Parse the JSON into a MovieReview
      review <- ZIO.fromEither(jsonStr.fromJson[MovieReview])
        .mapError(err => new RuntimeException(s"Failed to parse JSON: $err"))
      
      // Print the structured data
      _ <- printLine("Parsed Movie Review:")
      _ <- printLine(s"Title: ${review.title}")
      _ <- printLine(s"Year: ${review.year}")
      _ <- printLine(s"Rating: ${review.rating}")
      _ <- printLine(s"Review: ${review.review}")
    } yield ()
  }

  /**
   * Helper function to extract JSON from a string.
   * This handles cases where the LLM might include additional text before or after the JSON.
   */
  def extractJson(text: String): String = {
    val jsonPattern = """\{[\s\S]*\}""".r
    jsonPattern.findFirstIn(text).getOrElse("{}")
  }

  // Main program
  override def run: ZIO[Any, Throwable, Unit] = {
    // Create the program
    val program = for {
      // Print welcome message
      _ <- printLine("Welcome to ZIO LangChain Structured Output Example!")
      _ <- printLine("")
      
      // Get the LLM service
      llm <- ZIO.service[LLM]
      
      // Run the examples
      _ <- printLine("Running simple example...")
      _ <- simpleExample(llm)
    } yield ()
    
    // Provide the required services and run the program
    program.provide(
      // OpenAI LLM layer
      OpenAILLM.live,
      // OpenAI configuration layer
      ZLayer.succeed(
        OpenAIConfig(
          apiKey = sys.env.getOrElse("OPENAI_API_KEY", ""),
          model = sys.env.getOrElse("OPENAI_MODEL", "gpt-3.5-turbo"),
          temperature = 0.7,
          maxTokens = Some(2000)
        )
      )
    )
  }