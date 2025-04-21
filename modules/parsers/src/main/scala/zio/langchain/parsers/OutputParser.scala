package zio.langchain.parsers

import zio.*
import zio.json.*
import zio.langchain.core.errors.OutputParsingError
import zio.langchain.core.model.LLM
import zio.langchain.core.domain.ChatMessage

/**
 * OutputParser is a trait that defines how to parse LLM outputs into structured data.
 * It provides a generic interface for converting text outputs from LLMs into strongly typed Scala objects.
 *
 * @tparam T The type to parse the output into
 */
trait OutputParser[T]:
  /**
   * Parse the output text into the target type.
   *
   * @param text The text to parse
   * @return A ZIO effect that produces the parsed value or fails with an OutputParsingError
   */
  def parse(text: String): ZIO[Any, OutputParsingError, T]

  /**
   * Get the format instructions for this parser.
   * These instructions can be included in prompts to guide the LLM to produce outputs in the expected format.
   *
   * @return Format instructions as a string
   */
  def getFormatInstructions: String

  /**
   * Parse the output with retry capability.
   * If parsing fails, this method will attempt to fix the output using the provided LLM and retry.
   *
   * @param text The text to parse
   * @param llm The LLM to use for fixing parsing errors
   * @param maxRetries Maximum number of retry attempts
   * @return A ZIO effect that produces the parsed value or fails with an OutputParsingError
   */
  def parseWithRetry(
    text: String,
    llm: LLM,
    maxRetries: Int = 3
  ): ZIO[Any, OutputParsingError, T] =
    parse(text).catchSome {
      case error: OutputParsingError if maxRetries > 0 =>
        for
          fixedText <- fixOutput(text, error, llm)
          result <- parseWithRetry(fixedText, llm, maxRetries - 1)
        yield result
    }

  /**
   * Attempt to fix the output using the LLM.
   *
   * @param text The original text that failed to parse
   * @param error The parsing error
   * @param llm The LLM to use for fixing the output
   * @return A ZIO effect that produces the fixed text or fails with an OutputParsingError
   */
  private def fixOutput(
    text: String,
    error: OutputParsingError,
    llm: LLM
  ): ZIO[Any, OutputParsingError, String] =
    val prompt = s"""
      |I was trying to parse the following text into ${getFormatInstructions}:
      |
      |$text
      |
      |But I got the following error:
      |${error.getMessage}
      |
      |Please fix the text to match the expected format exactly.
      |""".stripMargin

    llm.complete(prompt)
      .mapError(e => OutputParsingError(e, "Failed to fix output format", Some(text)))

/**
 * Companion object for OutputParser.
 */
object OutputParser:
  /**
   * Creates a simple output parser that applies a function to the text.
   *
   * @param f The function to apply to the text
   * @param formatInstructions Format instructions for the parser
   * @return A new OutputParser
   */
  def apply[T](
    f: String => T,
    formatInstructions: String = "the expected format"
  ): OutputParser[T] =
    new OutputParser[T]:
      override def parse(text: String): ZIO[Any, OutputParsingError, T] =
        ZIO.attempt(f(text))
          .mapError(e => OutputParsingError(e, "Failed to parse output", Some(text)))

      override def getFormatInstructions: String = formatInstructions

  /**
   * Creates an output parser that uses ZIO JSON to parse the text into a case class.
   *
   * @param formatInstructions Format instructions for the parser
   * @param jsonOptions Optional JSON options for customizing parsing behavior
   * @return A new OutputParser that parses JSON into the specified type
   */
  def json[T: JsonDecoder: JsonEncoder: scala.reflect.ClassTag](
    formatInstructions: String = "JSON",
    jsonOptions: Option[JsonOptions] = None
  ): OutputParser[T] =
    new OutputParser[T]:
      override def parse(text: String): ZIO[Any, OutputParsingError, T] =
        ZIO.fromEither(
          jsonOptions match
            case Some(options) =>
              // Use the JsonDecoder directly since we can't pass our custom JsonOptions to fromJson
              val decoder = implicitly[JsonDecoder[T]]
              decoder.decodeJson(text)
            case None => text.fromJson[T]
        ).mapError(e => OutputParsingError(
          new RuntimeException(e),
          "Failed to parse JSON output",
          Some(text)
        ))

      override def getFormatInstructions: String =
        s"JSON with the following structure: ${scala.reflect.classTag[T].runtimeClass.getSimpleName}"

  /**
   * Creates an output parser that extracts structured data based on a regex pattern.
   *
   * @param pattern The regex pattern to use for extraction
   * @param transform A function that transforms the regex match groups into the target type
   * @param formatInstructions Format instructions for the parser
   * @return A new OutputParser that extracts data using regex
   */
  def regex[T](
    pattern: String,
    transform: Seq[String] => T,
    formatInstructions: String
  ): OutputParser[T] =
    new OutputParser[T]:
      private val compiledPattern = pattern.r

      override def parse(text: String): ZIO[Any, OutputParsingError, T] =
        ZIO.attempt {
          compiledPattern.findFirstMatchIn(text) match
            case Some(m) =>
              val groups = (0 to m.groupCount).map(i => m.group(i))
              transform(groups)
            case None =>
              throw new RuntimeException(s"Text does not match the expected pattern: $pattern")
        }.mapError(e => OutputParsingError(e, "Failed to parse output with regex", Some(text)))

      override def getFormatInstructions: String = formatInstructions

  /**
   * Creates an output parser that validates the parsed output against a schema.
   *
   * @param baseParser The base parser to use for initial parsing
   * @param validate A validation function that checks if the parsed value is valid
   * @param validationErrorMessage A function that generates an error message for validation failures
   * @return A new OutputParser that includes validation
   */
  def withValidation[T](
    baseParser: OutputParser[T],
    validate: T => Boolean,
    validationErrorMessage: T => String
  ): OutputParser[T] =
    new OutputParser[T]:
      override def parse(text: String): ZIO[Any, OutputParsingError, T] =
        for
          parsed <- baseParser.parse(text)
          _ <- ZIO.unless(validate(parsed)) {
            ZIO.fail(OutputParsingError(
              new RuntimeException(validationErrorMessage(parsed)),
              "Validation failed for parsed output",
              Some(text)
            ))
          }
        yield parsed

      override def getFormatInstructions: String =
        baseParser.getFormatInstructions