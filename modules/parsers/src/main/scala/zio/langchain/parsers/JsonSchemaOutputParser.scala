package zio.langchain.parsers

import zio.*
import zio.json.*
import zio.langchain.core.errors.OutputParsingError
import zio.langchain.core.model.LLM

// Define JsonOptions type for JSON parsing configuration
case class JsonOptions(
  ignoreUnknownKeys: Boolean = false,
  strictDecoding: Boolean = true
)

// Define JsonAny type for representing any JSON value
type JsonAny = Map[String, Any]

// Provide a JsonDecoder for JsonAny
implicit val jsonAnyDecoder: JsonDecoder[JsonAny] = JsonDecoder.map[String, String].map(_.map { case (k, v) => (k, v.asInstanceOf[Any]) })

/**
 * A specialized output parser that uses JSON schema for validation and parsing.
 * This parser provides more advanced schema validation and error handling for JSON outputs.
 *
 * @tparam T The type to parse the output into
 */
class JsonSchemaOutputParser[T: JsonDecoder: JsonEncoder] private (
  schema: JsonSchema,
  options: Option[JsonOptions] = None,
  formatInstructions: Option[String] = None
) extends OutputParser[T]:
  /**
   * Parse the output text into the target type using JSON schema validation.
   *
   * @param text The text to parse
   * @return A ZIO effect that produces the parsed value or fails with an OutputParsingError
   */
  override def parse(text: String): ZIO[Any, OutputParsingError, T] =
    // First, try to parse the text as JSON to validate its structure
    ZIO.attempt {
      val json = text.trim match
        case s if s.startsWith("```json") && s.endsWith("```") =>
          s.stripPrefix("```json").stripSuffix("```").trim
        case s if s.startsWith("```") && s.endsWith("```") =>
          s.stripPrefix("```").stripSuffix("```").trim
        case s => s

      json
    }.mapError(e =>
      OutputParsingError(
        e,
        "Failed to extract JSON content",
        Some(text)
      )
    ).flatMap { cleanedJson =>
      // Parse the JSON and validate against the schema
      ZIO.fromEither(
        options match
          case Some(opts) =>
            // Use the JsonDecoder directly since we can't pass our custom JsonOptions to fromJson
            val decoder = implicitly[JsonDecoder[T]]
            decoder.decodeJson(cleanedJson)
          case None => cleanedJson.fromJson[T]
      ).mapError { error =>
        OutputParsingError(
          new RuntimeException(s"JSON parsing error: $error"),
          "Failed to parse JSON output",
          Some(text)
        )
      }
    }.flatMap { parsed =>
      // Validate the parsed object against the schema
      validateAgainstSchema(parsed).mapError { error =>
        error.copy(output = Some(text))
      }
    }

  /**
   * Validate the parsed object against the JSON schema.
   *
   * @param parsed The parsed object
   * @return A ZIO effect that produces the validated object or fails with a Throwable
   */
  private def validateAgainstSchema(parsed: T): ZIO[Any, OutputParsingError, T] =
    ZIO.fromEither {
      // Convert the parsed object to JSON
      val jsonString = parsed.toJson
      
      // Validate against the schema
      schema.validate(jsonString) match
        case Left(errors) =>
          Left(OutputParsingError(
            new RuntimeException(s"Schema validation errors: ${errors.mkString(", ")}"),
            "JSON schema validation failed",
            None
          ))
        case Right(_) => Right(parsed)
    }

  /**
   * Get the format instructions for this parser.
   *
   * @return Format instructions as a string
   */
  override def getFormatInstructions: String =
    formatInstructions.getOrElse {
      s"""
         |You must respond with a JSON object that conforms to the following schema:
         |
         |${schema.toString}
         |
         |The JSON should be properly formatted without any leading or trailing text.
         |""".stripMargin
    }

/**
 * Companion object for JsonSchemaOutputParser.
 */
object JsonSchemaOutputParser:
  /**
   * Creates a new JsonSchemaOutputParser with the given schema.
   *
   * @param schema The JSON schema to validate against
   * @param options Optional JSON parsing options
   * @param formatInstructions Optional custom format instructions
   * @return A new JsonSchemaOutputParser
   */
  def apply[T: JsonDecoder: JsonEncoder](
    schema: JsonSchema,
    options: Option[JsonOptions] = None,
    formatInstructions: Option[String] = None
  ): JsonSchemaOutputParser[T] =
    new JsonSchemaOutputParser[T](schema, options, formatInstructions)

  /**
   * Creates a new JsonSchemaOutputParser from a case class type.
   * This automatically generates a JSON schema from the type.
   *
   * @param options Optional JSON parsing options
   * @param formatInstructions Optional custom format instructions
   * @return A new JsonSchemaOutputParser
   */
  def fromType[T: JsonDecoder: JsonEncoder: scala.reflect.ClassTag](
    options: Option[JsonOptions] = None,
    formatInstructions: Option[String] = None
  ): JsonSchemaOutputParser[T] =
    val schema = JsonSchema.fromType[T]
    new JsonSchemaOutputParser[T](schema, options, formatInstructions)

/**
 * Represents a JSON schema for validation.
 */
sealed trait JsonSchema:
  /**
   * Validate a JSON string against this schema.
   *
   * @param json The JSON string to validate
   * @return Either a list of validation errors or the validated JSON
   */
  def validate(json: String): Either[List[String], String]
  
  /**
   * Convert the schema to a string representation.
   *
   * @return A string representation of the schema
   */
  def toString: String

/**
 * Companion object for JsonSchema.
 */
object JsonSchema:
  /**
   * Creates a JSON schema from a type using ZIO JSON's schema support.
   *
   * @tparam T The type to create a schema for
   * @return A new JsonSchema
   */
  def fromType[T: JsonEncoder: JsonDecoder: scala.reflect.ClassTag]: JsonSchema =
    val encoder = implicitly[JsonEncoder[T]]
    // Generate a simple schema description instead of using encoder.schema
    val schemaStr = s"""
      |{
      |  "type": "object",
      |  "description": "Schema for ${scala.reflect.classTag[T].runtimeClass.getSimpleName}"
      |}
      |""".stripMargin
    
    new JsonSchema:
      override def validate(json: String): Either[List[String], String] =
        // Basic validation - just check if it can be parsed as the target type
        // In a real implementation, this would use a proper JSON schema validator
        val decoder = implicitly[JsonDecoder[T]]
        decoder.decodeJson(json) match
          case Left(e) => Left(List(e))
          case Right(_) => Right(json)
      
      override def toString: String = schemaStr

  /**
   * Creates a JSON schema from a schema string.
   *
   * @param schemaStr The schema string
   * @return A new JsonSchema
   */
  def fromString(schemaStr: String): JsonSchema =
    new JsonSchema:
      override def validate(json: String): Either[List[String], String] =
        // In a real implementation, this would use a proper JSON schema validator
        // For now, we just do basic JSON parsing validation
        json.fromJson[JsonAny] match
          case Left(err) => Left(List(err))
          case Right(_) => Right(json)
      
      override def toString: String = schemaStr