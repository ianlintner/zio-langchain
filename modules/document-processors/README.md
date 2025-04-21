# ZIO LangChain Document Processors

This module provides document processing capabilities for ZIO LangChain, with a focus on document chunking strategies for effective retrieval-augmented generation (RAG) systems.

## Document Chunking

Document chunking is the process of splitting documents into smaller pieces that can be processed by language models and vector databases. The choice of chunking strategy can significantly impact the performance of RAG systems.

### Available Chunking Strategies

The module provides several chunking strategies:

#### Character-based Chunking

Splits documents based on a fixed number of characters.

```scala
import zio.langchain.document.chunkers.DocumentChunkers

// Create a character-based chunker with a chunk size of 1000 characters and 200 characters overlap
val chunker = DocumentChunkers.byCharacterCount(chunkSize = 1000, chunkOverlap = 200)
```

#### Token-based Chunking

Splits documents based on a fixed number of tokens (estimated).

```scala
import zio.langchain.document.chunkers.DocumentChunkers

// Create a token-based chunker with a chunk size of 500 tokens and 50 tokens overlap
val chunker = DocumentChunkers.byTokenCount(chunkSize = 500, chunkOverlap = 50)
```

#### Separator-based Chunking

Splits documents at specific separator strings.

```scala
import zio.langchain.document.chunkers.DocumentChunkers

// Create a separator-based chunker that splits on newlines
val chunker = DocumentChunkers.bySeparator(separator = "\n", keepSeparator = false)
```

#### Sentence-based Chunking

Splits documents at sentence boundaries.

```scala
import zio.langchain.document.chunkers.DocumentChunkers

// Create a sentence-based chunker with a maximum of 5 sentences per chunk and 1 sentence overlap
val chunker = DocumentChunkers.bySentence(maxSentences = 5, overlapSentences = 1)
```

#### Paragraph-based Chunking

Splits documents at paragraph boundaries.

```scala
import zio.langchain.document.chunkers.DocumentChunkers

// Create a paragraph-based chunker with a maximum of 3 paragraphs per chunk and 1 paragraph overlap
val chunker = DocumentChunkers.byParagraph(maxParagraphs = 3, overlapParagraphs = 1)
```

#### Recursive Chunking

Applies multiple chunking strategies in sequence.

```scala
import zio.langchain.document.chunkers.DocumentChunkers

// Create a recursive chunker that first splits by paragraphs, then by sentences
val paragraphChunker = DocumentChunkers.byParagraph(maxParagraphs = 2)
val sentenceChunker = DocumentChunkers.bySentence(maxSentences = 3)
val recursiveChunker = DocumentChunkers.recursive(paragraphChunker, sentenceChunker)
```

#### Semantic Chunking

Splits documents based on semantic meaning (requires integration with an embedding model).

```scala
import zio.langchain.document.chunkers.DocumentChunkers

// Create a semantic chunker with a similarity threshold of 0.8
// and an initial chunker to create base chunks
val initialChunker = DocumentChunkers.byParagraph(maxParagraphs = 1)
val semanticChunker = DocumentChunkers.semantic(similarityThreshold = 0.8, initialChunker)
```

### Composing Chunkers

Chunkers can be composed using the `andThen` or `>>>` operator:

```scala
import zio.langchain.document.chunkers.DocumentChunkers

// Create a chunker that first splits by paragraphs, then by character count
val paragraphChunker = DocumentChunkers.byParagraph(maxParagraphs = 1)
val characterChunker = DocumentChunkers.byCharacterCount(chunkSize = 200)
val composedChunker = paragraphChunker >>> characterChunker
```

## Usage in RAG Systems

Document chunking is a critical component of RAG systems. Here's a simple example of how to use document chunkers in a RAG pipeline:

```scala
import zio.*
import zio.langchain.core.domain.Document
import zio.langchain.core.model.EmbeddingModel
import zio.langchain.document.chunkers.DocumentChunkers

// Load a document
val document = Document(
  id = "doc1",
  content = "Your document content here...",
  metadata = Map("source" -> "example.txt")
)

// Create a chunker
val chunker = DocumentChunkers.byParagraph(maxParagraphs = 2, overlapParagraphs = 1)

// Chunk the document
val program = for
  chunks <- chunker.chunk(document)
  _ <- ZIO.logInfo(s"Created ${chunks.size} chunks")
  
  // Get the embedding model
  embeddingModel <- ZIO.service[EmbeddingModel]
  
  // Create embeddings for the chunks
  embeddedChunks <- embeddingModel.embedDocuments(chunks)
  _ <- ZIO.logInfo("Created embeddings for all chunks")
  
  // Use the embedded chunks in a retriever
  // ...
yield ()
```

## Choosing the Right Chunking Strategy

The optimal chunking strategy depends on your specific use case:

- **Character-based chunking** is simple but may break semantic units.
- **Token-based chunking** is more relevant for LLMs but requires token estimation.
- **Sentence-based chunking** preserves sentence coherence but may create uneven chunks.
- **Paragraph-based chunking** preserves paragraph coherence and is often a good default.
- **Recursive chunking** provides more fine-grained control over chunk size and coherence.
- **Semantic chunking** can create more meaningful chunks but requires an embedding model.

Experiment with different strategies to find the one that works best for your specific documents and retrieval needs.