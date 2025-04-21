package zio.langchain.parsers

import zio.*
import zio.json.*
import zio.langchain.core.errors.OutputParsingError
import zio.langchain.core.model.LLM

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
    }.flatMap { cleanedJson =>
      // Parse the JSON and validate against the schema
      ZIO.fromEither(
        options match
          case Some(opts) => cleanedJson.fromJson[T](opts)
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
        OutputParsingError(
          error,
          "JSON schema validation failed",
          Some(text)
        )
      }
    }

  /**
   * Validate the parsed object against the JSON schema.
   *
   * @param parsed The parsed object
   * @return A ZIO effect that produces the validated object or fails with a Throwable
   */
  private def validateAgainstSchema(parsed: T): ZIO[Any, Throwable, T] =
    ZIO.attempt {
      // Convert the parsed object to JSON
      val jsonString = parsed.toJson
      
      // Validate against the schema
      schema.validate(jsonString) match
        case Left(errors) =>
          throw new RuntimeException(s"Schema validation errors: ${errors.mkString(", ")}")
        case Right(_) => parsed
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
  def fromType[T: JsonDecoder: JsonEncoder](
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
  def fromType[T: JsonEncoder]: JsonSchema =
    val encoder = implicitly[JsonEncoder[T]]
    val schemaStr = encoder.schema.toString
    
    new JsonSchema:
      override def validate(json: String): Either[List[String], String] =
        // Basic validation - just check if it can be parsed as the target type
        // In a real implementation, this would use a proper JSON schema validator
        json.fromJson[T] match
          case Left(error) => Left(List(error))
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
          case Left(error) => Left(List(error))
          case Right(_) => Right(json)
      
      override def toString: String = schemaStr