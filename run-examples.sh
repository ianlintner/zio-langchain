#!/bin/bash

# ZIO LangChain Examples Runner
# This script allows users to easily run the example applications

set -e

# Colors for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Check if OpenAI API key is set
check_api_key() {
  if [ -z "${OPENAI_API_KEY}" ]; then
    echo -e "${RED}Error: OPENAI_API_KEY environment variable is not set.${NC}"
    echo -e "Please set it using: ${BLUE}export OPENAI_API_KEY=\"your-key-here\"${NC}"
    exit 1
  fi
}

# Print the header
print_header() {
  echo -e "${GREEN}========================================${NC}"
  echo -e "${GREEN}      ZIO LangChain Examples Runner     ${NC}"
  echo -e "${GREEN}========================================${NC}"
  echo
}

# Print available examples
print_examples() {
  echo -e "${BLUE}Available Examples:${NC}"
  echo -e "  ${YELLOW}1${NC}. SimpleChat       - Basic chat interaction with LLM"
  echo -e "  ${YELLOW}2${NC}. SimpleRAG        - Basic Retrieval Augmented Generation"
  echo -e "  ${YELLOW}3${NC}. AdvancedChat     - Advanced chat with memory and tools"
  echo -e "  ${YELLOW}4${NC}. EnhancedRAG      - Enhanced RAG with better retrieval"
  echo -e "  ${YELLOW}5${NC}. SimpleAgent      - Simple agent implementation"
  echo
}

# Print usage help
print_help() {
  print_header
  print_examples
  echo -e "${BLUE}Usage:${NC}"
  echo -e "  ./run-examples.sh [example-name]"
  echo
  echo -e "${BLUE}Examples:${NC}"
  echo -e "  ./run-examples.sh SimpleChat"
  echo -e "  ./run-examples.sh EnhancedRAG"
  echo
  echo -e "${BLUE}Environment Variables:${NC}"
  echo -e "  OPENAI_API_KEY     - Your OpenAI API key (required)"
  echo -e "  OPENAI_MODEL       - Model to use (default: gpt-3.5-turbo)"
  echo -e "  OPENAI_TEMPERATURE - Temperature setting (default: 0.7)"
  echo
}

# Run the specified example
run_example() {
  local example="$1"
  check_api_key
  
  echo -e "${GREEN}Running ${example}...${NC}"
  echo -e "${BLUE}Press Ctrl+C to stop the example${NC}"
  echo
  
  # Set default model if not specified
  if [ -z "${OPENAI_MODEL}" ]; then
    export OPENAI_MODEL="gpt-3.5-turbo"
  fi
  
  # Set default temperature if not specified
  if [ -z "${OPENAI_TEMPERATURE}" ]; then
    export OPENAI_TEMPERATURE="0.7"
  fi
  
  case "$example" in
    "SimpleChat")
      sbt "examples/runMain zio.langchain.examples.SimpleChat"
      ;;
    "SimpleRAG")
      sbt "examples/runMain zio.langchain.examples.SimpleRAG"
      ;;
    "AdvancedChat")
      sbt "examples/runMain zio.langchain.examples.AdvancedChat"
      ;;
    "EnhancedRAG")
      sbt "examples/runMain zio.langchain.examples.EnhancedRAG"
      ;;
    "SimpleAgent")
      sbt "examples/runMain zio.langchain.examples.SimpleAgent"
      ;;
    *)
      echo -e "${RED}Unknown example: ${example}${NC}"
      print_examples
      exit 1
      ;;
  esac
}

# Interactive mode if no arguments provided
run_interactive() {
  print_header
  print_examples
  
  echo -e "${BLUE}Enter example number (or 'q' to quit):${NC}"
  read -p "> " selection
  
  case "$selection" in
    "1")
      run_example "SimpleChat"
      ;;
    "2")
      run_example "SimpleRAG"
      ;;
    "3")
      run_example "AdvancedChat"
      ;;
    "4")
      run_example "EnhancedRAG"
      ;;
    "5")
      run_example "SimpleAgent"
      ;;
    "q"|"Q"|"quit"|"exit")
      echo -e "${GREEN}Goodbye!${NC}"
      exit 0
      ;;
    *)
      echo -e "${RED}Invalid selection. Please try again.${NC}"
      run_interactive
      ;;
  esac
}

# Main entry point logic
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
  print_help
elif [ -z "$1" ]; then
  run_interactive
else
  run_example "$1"
fi