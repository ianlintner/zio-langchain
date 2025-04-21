package zio.langchain.core.document

import zio.*
import zio.stream.ZStream

import zio.langchain.core.domain.Document
import zio.langchain.core.errors.DocumentLoadingError

/**
 * Interface for document loaders.
 * Document loaders are responsible for loading documents from various sources.
 */
trait DocumentLoader:
  /**
   * Loads documents from the source.
   *
   * @return A ZStream that produces documents or fails with a DocumentLoadingError
   */
  def load: ZStream[Any, DocumentLoadingError, Document]
  
  /**
   * Loads documents from the source and collects them into a sequence.
   *
   * @return A ZIO effect that produces a sequence of documents or fails with a DocumentLoadingError
   */
  def loadAll: ZIO[Any, DocumentLoadingError, Seq[Document]] =
    load.runCollect.map(_.toSeq)
  
  /**
   * Transforms the documents produced by this loader using the given function.
   *
   * @param f The function to apply to each document
   * @return A new document loader that applies the function to each document
   */
  def map(f: Document => Document): DocumentLoader =
    DocumentLoader.Map(this, f)
  
  /**
   * Filters the documents produced by this loader using the given predicate.
   *
   * @param p The predicate to apply to each document
   * @return A new document loader that only includes documents that satisfy the predicate
   */
  def filter(p: Document => Boolean): DocumentLoader =
    DocumentLoader.Filter(this, p)

/**
 * Companion object for DocumentLoader.
 */
object DocumentLoader:
  /**
   * Creates a document loader from a single document.
   *
   * @param document The document to load
   * @return A new document loader that produces the document
   */
  def single(document: Document): DocumentLoader =
    new DocumentLoader:
      override def load: ZStream[Any, DocumentLoadingError, Document] =
        ZStream.succeed(document)
  
  /**
   * Creates a document loader from a sequence of documents.
   *
   * @param documents The documents to load
   * @return A new document loader that produces the documents
   */
  def fromSeq(documents: Seq[Document]): DocumentLoader =
    new DocumentLoader:
      override def load: ZStream[Any, DocumentLoadingError, Document] =
        ZStream.fromIterable(documents)
  
  /**
   * Creates a document loader from a ZIO effect that produces a document.
   *
   * @param effect The effect that produces a document
   * @return A new document loader that produces the document
   */
  def fromEffect(effect: ZIO[Any, Throwable, Document]): DocumentLoader =
    new DocumentLoader:
      override def load: ZStream[Any, DocumentLoadingError, Document] =
        ZStream.fromZIO(
          effect.mapError(e => DocumentLoadingError(e))
        )
  
  /**
   * Creates a document loader from a ZIO effect that produces a sequence of documents.
   *
   * @param effect The effect that produces a sequence of documents
   * @return A new document loader that produces the documents
   */
  def fromEffectSeq(effect: ZIO[Any, Throwable, Seq[Document]]): DocumentLoader =
    new DocumentLoader:
      override def load: ZStream[Any, DocumentLoadingError, Document] =
        ZStream.fromIterableZIO(
          effect.mapError(e => DocumentLoadingError(e))
        )
  
  /**
   * A document loader that transforms documents using a function.
   *
   * @param source The source document loader
   * @param f The function to apply to each document
   */
  private case class Map(
    source: DocumentLoader,
    f: Document => Document
  ) extends DocumentLoader:
    override def load: ZStream[Any, DocumentLoadingError, Document] =
      source.load.map(f)
  
  /**
   * A document loader that filters documents using a predicate.
   *
   * @param source The source document loader
   * @param p The predicate to apply to each document
   */
  private case class Filter(
    source: DocumentLoader,
    p: Document => Boolean
  ) extends DocumentLoader:
    override def load: ZStream[Any, DocumentLoadingError, Document] =
      source.load.filter(p)