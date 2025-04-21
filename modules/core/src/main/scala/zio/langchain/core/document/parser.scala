package zio.langchain.core.document

import zio.*

import zio.langchain.core.domain.Document
import zio.langchain.core.errors.DocumentParsingError

/**
 * Interface for document parsers.
 * Document parsers are responsible for parsing documents into smaller chunks or extracting specific information.
 */
trait DocumentParser:
  /**
   * Parses a document into multiple documents.
   * This is typically used for splitting a document into smaller chunks.
   *
   * @param document The document to parse
   * @return A ZIO effect that produces a sequence of documents or fails with a DocumentParsingError
   */
  def parse(document: Document): ZIO[Any, DocumentParsingError, Seq[Document]]
  
  /**
   * Parses multiple documents.
   *
   * @param documents The documents to parse
   * @return A ZIO effect that produces a sequence of documents or fails with a DocumentParsingError
   */
  def parseAll(documents: Seq[Document]): ZIO[Any, DocumentParsingError, Seq[Document]] =
    ZIO.foreachPar(documents)(parse).map(_.flatten)
  
  /**
   * Composes this parser with another parser, where the output of this parser is passed as input to the next parser.
   *
   * @param next The next parser to compose with
   * @return A new parser that represents the composition of this parser with the next parser
   */
  def andThen(next: DocumentParser): DocumentParser =
    DocumentParser.Sequence(this, next)
  
  /**
   * Alias for andThen.
   */
  def >>>(next: DocumentParser): DocumentParser =
    andThen(next)

/**
 * Companion object for DocumentParser.
 */
object DocumentParser:
  /**
   * Creates a document parser that applies a function to each document.
   *
   * @param f The function to apply to each document
   * @return A new document parser that applies the function to each document
   */
  def apply(f: Document => Seq[Document]): DocumentParser =
    new DocumentParser:
      override def parse(document: Document): ZIO[Any, DocumentParsingError, Seq[Document]] =
        ZIO.attempt(f(document))
          .mapError(e => DocumentParsingError(e))
  
  /**
   * Creates a document parser that applies a ZIO effect to each document.
   *
   * @param f The effect to apply to each document
   * @return A new document parser that applies the effect to each document
   */
  def fromEffect(f: Document => ZIO[Any, Throwable, Seq[Document]]): DocumentParser =
    new DocumentParser:
      override def parse(document: Document): ZIO[Any, DocumentParsingError, Seq[Document]] =
        f(document).mapError(e => DocumentParsingError(e))
  
  /**
   * Creates a document parser that doesn't modify the document.
   *
   * @return A new document parser that returns the document unchanged
   */
  def identity: DocumentParser =
    new DocumentParser:
      override def parse(document: Document): ZIO[Any, DocumentParsingError, Seq[Document]] =
        ZIO.succeed(Seq(document))
  
  /**
   * A document parser that composes two parsers.
   *
   * @param first The first parser
   * @param second The second parser
   */
  private case class Sequence(
    first: DocumentParser,
    second: DocumentParser
  ) extends DocumentParser:
    override def parse(document: Document): ZIO[Any, DocumentParsingError, Seq[Document]] =
      for
        firstResults <- first.parse(document)
        secondResults <- ZIO.foreachPar(firstResults)(second.parse)
      yield secondResults.flatten

/**
 * Common document parsing strategies.
 */
object DocumentParsers:
  /**
   * Creates a document parser that splits documents by character count.
   *
   * @param chunkSize The maximum size of each chunk in characters
   * @param chunkOverlap The number of characters to overlap between chunks
   * @return A new document parser that splits documents by character count
   */
  def byCharacterCount(chunkSize: Int, chunkOverlap: Int = 0): DocumentParser =
    new DocumentParser:
      override def parse(document: Document): ZIO[Any, DocumentParsingError, Seq[Document]] =
        ZIO.attempt {
          val text = document.content
          val chunks = splitTextByCharacterCount(text, chunkSize, chunkOverlap)
          chunks.zipWithIndex.map { case (chunk, i) =>
            Document(
              id = s"${document.id}-chunk-$i",
              content = chunk,
              metadata = document.metadata + ("chunk" -> i.toString)
            )
          }
        }.mapError(e => DocumentParsingError(e))
      
      private def splitTextByCharacterCount(text: String, chunkSize: Int, chunkOverlap: Int): Seq[String] =
        if text.length <= chunkSize then
          Seq(text)
        else
          val chunks = Seq.newBuilder[String]
          var start = 0
          while start < text.length do
            val end = math.min(start + chunkSize, text.length)
            chunks += text.substring(start, end)
            start += (chunkSize - chunkOverlap)
          chunks.result()
  
  /**
   * Creates a document parser that splits documents by separator.
   *
   * @param separator The separator to split on
   * @param keepSeparator Whether to keep the separator in the chunks
   * @return A new document parser that splits documents by separator
   */
  def bySeparator(separator: String, keepSeparator: Boolean = false): DocumentParser =
    new DocumentParser:
      override def parse(document: Document): ZIO[Any, DocumentParsingError, Seq[Document]] =
        ZIO.attempt {
          val text = document.content
          val chunks = text.split(separator, -1).toSeq
          val finalChunks = if keepSeparator then
            chunks.map(_ + separator)
          else
            chunks
          
          finalChunks.zipWithIndex.map { case (chunk, i) =>
            Document(
              id = s"${document.id}-chunk-$i",
              content = chunk,
              metadata = document.metadata + ("chunk" -> i.toString)
            )
          }
        }.mapError(e => DocumentParsingError(e))