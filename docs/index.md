---
title: ZIO LangChain Documentation
author: ZIO LangChain Team
date: April 21, 2025
version: 0.1.0
---

# ZIO LangChain Documentation

A comprehensive Scala 3 library that provides a ZIO-based langchain, offering a purely functional effect based, type-safe API for building LLM-powered applications.

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
- [Core Concepts](#core-concepts)
- [Components](#components)
- [Examples](#examples)
- [API Reference](#api-reference)
- [Design Documentation](#design-documentation)

## Overview

ZIO LangChain is a Scala 3 library that wraps langchain4j in a purely functional, ZIO-based API. It provides a type-safe, composable way to build LLM-powered applications with proper resource management via ZIO.

### Key Features

- **Pure Functional**: All operations are represented as ZIO effects
- **Type-Safe**: Leverage Scala 3's type system for safer code
- **Composable**: Build complex workflows by composing simple components
- **Resource-Safe**: Proper resource management via ZIO
- **Streaming Support**: Stream tokens and process large documents efficiently
- **Comprehensive**: Support for all major langchain4j features

## Getting Started

- [Installation](getting-started/installation.md)
- [Configuration](getting-started/configuration.md)
- [Quick Start Guide](getting-started/quickstart.md)

## Core Concepts

- [LLM Integration](core-concepts/llm-integration.md)
- [Embeddings](core-concepts/embeddings.md)
- [Memory](core-concepts/memory.md)
- [Documents](core-concepts/documents.md)
- [Retrieval](core-concepts/retrieval.md)
- [Chains](core-concepts/chains.md)
- [Agents](core-concepts/agents.md)
- [Tools](core-concepts/tools.md)

## Components

- [Models](components/models/index.md)
- [Embeddings](components/embeddings/index.md)
- [Memory](components/memory/index.md)
- [Document Loaders](components/document-loaders/index.md)
- [Document Parsers](components/document-parsers/index.md)
- [Retrievers](components/retrievers/index.md)
- [Chains](components/chains/index.md)
- [Agents](components/agents/index.md)
- [Tools](components/tools/index.md)

## Examples

- [Simple Chat](examples/simple-chat.md)
- [Simple RAG](examples/simple-rag.md)
- [Advanced Chat](examples/advanced-chat.md)
- [Enhanced RAG](examples/enhanced-rag.md)
- [Simple Agent](examples/simple-agent.md)

## API Reference

- [API Documentation](api/index.md)

## Design Documentation

- [Architecture](design/architecture.md)
- [Implementation Strategy](design/implementation.md)