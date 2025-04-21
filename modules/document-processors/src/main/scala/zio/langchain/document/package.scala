package zio.langchain

/**
 * Document processing module for ZIO LangChain.
 * This package provides document processing capabilities, including document chunking strategies.
 */
package object document:
  /**
   * Re-exports from the chunkers package.
   */
  type DocumentChunker = chunkers.DocumentChunker
  type ChunkerConfig = chunkers.ChunkerConfig
  type DocumentChunkingError = chunkers.DocumentChunkingError
  
  /**
   * Re-exports specific chunker configurations.
   */
  type CharacterChunkerConfig = chunkers.CharacterChunkerConfig
  type TokenChunkerConfig = chunkers.TokenChunkerConfig
  type SeparatorChunkerConfig = chunkers.SeparatorChunkerConfig
  type SentenceChunkerConfig = chunkers.SentenceChunkerConfig
  type ParagraphChunkerConfig = chunkers.ParagraphChunkerConfig
  type RecursiveChunkerConfig = chunkers.RecursiveChunkerConfig
  type SemanticChunkerConfig = chunkers.SemanticChunkerConfig
  
  /**
   * Re-exports specific chunker implementations.
   */
  type CharacterChunker = chunkers.CharacterChunker
  type TokenChunker = chunkers.TokenChunker
  type SeparatorChunker = chunkers.SeparatorChunker
  type SentenceChunker = chunkers.SentenceChunker
  type ParagraphChunker = chunkers.ParagraphChunker
  type RecursiveChunker = chunkers.RecursiveChunker
  type SemanticChunker = chunkers.SemanticChunker
  
  /**
   * Re-exports the DocumentChunkers factory object.
   */
  val DocumentChunkers = chunkers.DocumentChunkers