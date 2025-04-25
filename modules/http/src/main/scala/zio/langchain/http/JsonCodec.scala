package zio.langchain.http

import zio.json.{JsonDecoder, JsonEncoder}

/**
 * Provider for JSON serialization and deserialization.
 */
trait JsonCodecProvider:
  /**
   * Encode a value to a JSON string.
   *
   * @param value The value to encode
   * @param encoder The encoder for the value type
   * @return The JSON string
   */
  def encode[A](value: A)(using encoder: JsonEncoder[A]): String
  
  /**
   * Decode a JSON string to a value.
   *
   * @param json The JSON string to decode
   * @param decoder The decoder for the value type
   * @return Either a decoded value or an error message
   */
  def decode[A](json: String)(using decoder: JsonDecoder[A]): Either[String, A]

/**
 * Implementation of JsonCodecProvider using zio-json.
 */
object ZioJsonProvider extends JsonCodecProvider:
  import zio.json.*
  
  override def encode[A](value: A)(using encoder: JsonEncoder[A]): String =
    value.toJson
    
  override def decode[A](json: String)(using decoder: JsonDecoder[A]): Either[String, A] =
    json.fromJson[A]