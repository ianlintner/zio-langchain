---
title: Documents
author: ZIO LangChain Team
date: April 21, 2025
version: 0.1.0
---

# Documents

This document explains the document abstractions in ZIO LangChain - how they represent text content, their structure, and how to work with them effectively.

## Table of Contents

- [Introduction](#introduction)
- [Document Representation](#document-representation)
- [Document Loaders](#document-loaders)
- [Document Parsers](#document-parsers)
- [Working with Documents](#working-with-documents)
- [Best Practices](#best-practices)

## Introduction

Documents are a fundamental abstraction in ZIO LangChain, representing text data with associated metadata. They serve as the primary unit of information in:

- Retrieval-Augmented Generation (RAG) systems
- Knowledge bases and document stores
- Text processing and analysis pipelines
- Information extraction workflows

The document abstraction provides a consistent way to work with textual data regardless of its source or format, allowing for seamless integration of various data sources into LLM applications.

## Document Representation

### Core Document Type

The foundation of document handling in ZIO LangChain is the `Document` case class:

```scala
case class Document(
  id: String,
  content: String,
  metadata: Map[String, String] = Map.empty
)
```

This simple yet powerful representation includes:

- **id**: A unique identifier for the document
- **content**: The actual text content
- **metadata**: A flexible map of key-value pairs for additional information

### Metadata

The metadata field can store various types of information about the document:

- **Source information**: File path, URL, database reference
- **Structural information**: Page number, section, chapter
- **Temporal information**: Creation date, last modified date
- **Authorship**: Author, organization, department
- **Classification**: Tags, categories, topics
- **Processing information**: Chunking strategy, embedding information

Example document with metadata:

```scala
val document = Document(
  id = "doc-001",
  content = "ZIO LangChain is a Scala wrapper around langchain4j with ZIO integration.",
  metadata = Map(
    "source" -> "documentation",
    "section" -> "introduction",
    "author" -> "ZIO Team",
    "date" -> "2025-04-01",
    "language" -> "en-US"
  )
)
```

## Document Loaders

Document loaders are responsible for loading documents from various sources into the `Document` representation.

### Core Document Loader Interface

ZIO LangChain provides a `DocumentLoader` trait:

```scala
trait DocumentLoader:
  def load: ZStream[Any, Throwable, Document]
```

This interface returns a ZStream of documents, enabling efficient processing of large document collections through streaming.

### Common Document Loaders

#### Text File Loader

```scala
class TextFileLoader(path: Path) extends DocumentLoader:
  override def load: ZStream[Any, Throwable, Document] =
    ZStream.fromZIO(
      ZIO.attemptBlockingIO {
        val content = Files.readString(path)
        Document(
          id = path.toString,
          content = content,
          metadata = Map("source" -> path.toString)
        )
      }
    )
```

#### PDF Loader

```scala
class PDFLoader(path: Path) extends DocumentLoader:
  override def load: ZStream[Any, Throwable, Document] =
    ZStream.fromZIO(
      ZIO.attemptBlockingIO {
        // Use langchain4j's PDF loader under the hood
        val pdfDocument = new dev.langchain4j.data.document.loader.FileSystemPdfDocumentLoader(path.toFile).load()
        Document(
          id = path.toString,
          content = pdfDocument.text(),
          metadata = pdfDocument.metadata().asScala.toMap
        )
      }
    )
```

#### Directory Loader

```scala
class DirectoryLoader(
  directory: Path,
  fileExtensions: Set[String] = Set(".txt", ".md", ".pdf")
) extends DocumentLoader:
  override def load: ZStream[Any, Throwable, Document] =
    ZStream.fromZIO(
      ZIO.attemptBlockingIO {
        Files.walk(directory)
          .filter(path => Files.isRegularFile(path))
          .filter(path => 
            fileExtensions.exists(ext => path.toString.endsWith(ext))
          )
          .toList.asScala
      }
    ).flatMap { filePath =>
      val loader = if (filePath.toString.endsWith(".pdf")) {
        new PDFLoader(filePath)
      } else {
        new TextFileLoader(filePath)
      }
      loader.load
    }
```

#### Web Loader

```scala
class WebLoader(url: String) extends DocumentLoader:
  override def load: ZStream[Any, Throwable, Document] =
    ZStream.fromZIO(
      ZIO.attemptBlockingIO {
        // Implementation to fetch content from a URL
        val content = /* fetch content */
        Document(
          id = url,
          content = content,
          metadata = Map("source" -> url)
        )
      }
    )
```

## Document Parsers

Document parsers transform documents, typically by splitting them into smaller chunks for more effective processing.

### Core Document Parser Interface

```scala
trait DocumentParser:
  def splitDocument(document: Document): Seq[Document]
```

### Common Document Parsers

#### Character Text Splitter

```scala
class CharacterTextSplitter(
  chunkSize: Int = 1000,
  chunkOverlap: Int = 200,
  separator: String = "\n\n"
) extends DocumentParser:
  override def splitDocument(document: Document): Seq[Document] =
    val text = document.content
    val chunks = splitText(text)
    chunks.zipWithIndex.map { case (chunk, index) =>
      Document(
        id = s"${document.id}-chunk-$index",
        content = chunk,
        metadata = document.metadata + ("chunk" -> index.toString)
      )
    }
    
  private def splitText(text: String): Seq[String] =
    // Implementation of chunking logic
```

#### Recursive Character Text Splitter

```scala
class RecursiveCharacterTextSplitter(
  chunkSize: Int = 1000,
  chunkOverlap: Int = 200,
  separators: Seq[String] = Seq("\n\n", "\n", ". ", " ", "")
) extends DocumentParser:
  override def splitDocument(document: Document): Seq[Document] =
    // Implementation that recursively tries different separators
```

#### Code Text Splitter

```scala
class CodeTextSplitter(
  language: String,
  chunkSize: Int = 1000,
  chunkOverlap: Int = 200
) extends DocumentParser:
  override def splitDocument(document: Document): Seq[Document] =
    // Language-specific chunking (by function, class, etc.)
```

## Working with Documents

### Loading and Processing Documents

```scala
import zio.*
import zio.langchain.core.document.*

val program = for
  // Create a document loader
  loader = new DirectoryLoader(Path.of("docs"))
  
  // Load documents
  documents <- loader.load.runCollect.map(_.toSeq)
  
  // Create a parser
  parser = new CharacterTextSplitter()
  
  // Split documents into chunks
  chunks = documents.flatMap(parser.splitDocument)
  
  // Process chunks (e.g., create embeddings)
  _ <- ZIO.foreach(chunks) { chunk =>
    Console.printLine(s"Processing chunk: ${chunk.id}")
  }
yield chunks
```

### Filtering Documents

```scala
// Filter documents by metadata
def filterByMetadata(
  documents: Seq[Document],
  key: String,
  value: String
): Seq[Document] =
  documents.filter(_.metadata.get(key).contains(value))

// Filter by content
def filterByContent(
  documents: Seq[Document],
  predicate: String => Boolean
): Seq[Document] =
  documents.filter(doc => predicate(doc.content))
```

### Transforming Documents

```scala
// Transform document content
def transformContent(
  documents: Seq[Document],
  transformer: String => String
): Seq[Document] =
  documents.map(doc => doc.copy(content = transformer(doc.content)))

// Add metadata
def addMetadata(
  documents: Seq[Document],
  key: String,
  value: String
): Seq[Document] =
  documents.map(doc => doc.copy(metadata = doc.metadata + (key -> value)))
```

## Best Practices

### Document Loading

1. **Streaming**: Use ZStreams for large document collections to avoid memory issues
   ```scala
   loader.load
     .tap(doc => processDocument(doc))
     .runDrain
   ```

2. **Error Handling**: Implement robust error handling for document loading
   ```scala
   loader.load
     .catchSome {
       case e: IOException => ZStream.succeed(createErrorDocument(e))
     }
   ```

3. **Parallelism**: Load and process documents in parallel when appropriate
   ```scala
   ZIO.foreachPar(paths)(path => new PDFLoader(path).load.runCollect)
   ```

### Document Chunking

1. **Chunk Size**: Balance chunk size based on your use case
   - Too large: May dilute relevance in retrieval
   - Too small: May lose important context

2. **Semantic Boundaries**: Try to split at semantic boundaries (paragraphs, sections)
   ```scala
   new RecursiveCharacterTextSplitter(
     separators = Seq("\n## ", "\n### ", "\n\n", "\n", ". ")
   )
   ```

3. **Overlap**: Include some overlap between chunks for context continuity
   ```scala
   new CharacterTextSplitter(chunkSize = 1000, chunkOverlap = 200)
   ```

4. **Preserve Metadata**: Maintain source information in chunk metadata
   ```scala
   chunks.map(chunk => chunk.copy(
     metadata = chunk.metadata + ("originalId" -> document.id)
   ))
   ```

### Document Management

1. **Unique IDs**: Ensure documents have unique, meaningful IDs
   ```scala
   val hasher = java.security.MessageDigest.getInstance("SHA-256")
   val documentId = bytesToHex(hasher.digest(content.getBytes))
   ```

2. **Rich Metadata**: Include detailed metadata for better filtering
   ```scala
   val metadata = Map(
     "source" -> source,
     "created" -> timestamp,
     "language" -> detectLanguage(content),
     "word_count" -> wordCount.toString
   )
   ```

3. **Content Preprocessing**: Consider preprocessing document content
   ```scala
   def preprocessContent(content: String): String =
     content
       .replaceAll("\\s+", " ") // Normalize whitespace
       .trim
   ```

4. **Document Deduplication**: Implement deduplication strategies
   ```scala
   def deduplicateDocuments(documents: Seq[Document]): Seq[Document] =
     documents.groupBy(_.content).map(_._2.head).toSeq