package zio.langchain.http.streaming

import zio.*
import zio.stream.*
import zio.json.*
import zio.langchain.http.{JsonCodecProvider, SerializationError}

/**
 * Utilities for handling streaming HTTP responses.
 */
object HttpStreaming:
  /**
   * Creates a pipeline for processing JSON lines from a stream.
   */
  object JsonStreamingPipeline:
    /**
     * Creates a pipeline that processes a stream of bytes as newline-delimited JSON.
     *
     * @param jsonCodecProvider The JSON codec provider to use for decoding
     * @param decoder The JSON decoder for the element type
     * @return A ZPipeline that transforms a stream of bytes into a stream of decoded elements
     */
    def jsonLines[A](jsonCodecProvider: JsonCodecProvider)(using decoder: JsonDecoder[A]): ZPipeline[Any, SerializationError, Byte, A] =
      (ZPipeline.utf8Decode.mapError(e =>
        SerializationError(s"Failed to decode UTF-8: ${e.getMessage}", Some(e))
      ) >>>
      ZPipeline.splitLines >>>
      ZPipeline.mapZIO((line: String) =>
        if (line.trim.isEmpty) {
          ZIO.succeed(None)
        } else {
          ZIO.fromEither(jsonCodecProvider.decode[A](line))
            .mapError(err => SerializationError(s"Failed to decode JSON: $err", None))
            .map(Some(_))
        }
      ) >>>
      ZPipeline.collect[Option[A], A] { case Some(value) => value })

    /**
     * Creates a pipeline that processes a stream of bytes as a JSON array.
     *
     * @param jsonCodecProvider The JSON codec provider to use for decoding
     * @param decoder The JSON decoder for the element type
     * @return A ZPipeline that transforms a stream of bytes into a stream of decoded elements
     */
    def jsonArray[A](jsonCodecProvider: JsonCodecProvider)(using decoder: JsonDecoder[A]): ZPipeline[Any, SerializationError, Byte, A] =
      (ZPipeline.utf8Decode.mapError(e =>
        SerializationError(s"Failed to decode UTF-8: ${e.getMessage}", Some(e))
      ) >>>
      ZPipeline.mapZIO((jsonStr: String) =>
        ZIO.fromEither(jsonCodecProvider.decode[List[A]](jsonStr))
          .mapError(err => SerializationError(s"Failed to decode JSON array: $err", None))
      ) >>>
      ZPipeline.flattenIterables)

    /**
     * Creates a pipeline that processes a stream of bytes as a Server-Sent Events (SSE) stream.
     *
     * @param jsonCodecProvider The JSON codec provider to use for decoding
     * @param decoder The JSON decoder for the element type
     * @param dataFieldName The name of the field in the SSE data that contains the JSON payload
     * @return A ZPipeline that transforms a stream of bytes into a stream of decoded elements
     */
    def serverSentEvents[A](
      jsonCodecProvider: JsonCodecProvider,
      dataFieldName: String = "data"
    )(using decoder: JsonDecoder[A]): ZPipeline[Any, SerializationError, Byte, A] =
      (ZPipeline.utf8Decode.mapError(e =>
        SerializationError(s"Failed to decode UTF-8: ${e.getMessage}", Some(e))
      ) >>>
      ZPipeline.splitLines >>>
      ZPipeline.scan[String, Option[String]](Option.empty[String]) { (acc, line) =>
        if (line.startsWith(s"$dataFieldName: ")) {
          Some(line.substring(dataFieldName.length + 2))
        } else if (line.isEmpty && acc.isDefined) {
          // Empty line marks the end of an event
          acc
        } else {
          None
        }
      } >>>
      ZPipeline.collect[Option[String], String] { case Some(data) => data } >>>
      ZPipeline.mapZIO((jsonData: String) =>
        ZIO.fromEither(jsonCodecProvider.decode[A](jsonData))
          .mapError(err => SerializationError(s"Failed to decode SSE data: $err", None))
      ))