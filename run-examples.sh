#!/bin/bash

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to show usage
show_usage() {
  echo -e "${CYAN}ZIO LangChain Examples Runner${NC}"
  echo ""
  echo "This script helps run the example applications from the ZIO LangChain project."
  echo ""
  echo -e "${YELLOW}Usage:${NC}"
  echo "  ./run-examples.sh [EXAMPLE_NAME]"
  echo ""
  echo -e "${YELLOW}Available examples:${NC}"
  echo "  simple-chat    - Run the SimpleChat example (basic chat with OpenAI)"
  echo "  simple-rag     - Run the SimpleRAG example (basic retrieval-augmented generation)"
  echo "  advanced-chat  - Run the AdvancedChat example (advanced chat with memory)"
  echo "  enhanced-rag   - Run the EnhancedRAG example (enhanced RAG with embeddings)"
  echo "  simple-agent   - Run the SimpleAgent example (basic agent with tools)"
  echo "  list           - List all available examples"
  echo "  help           - Show this help message"
  echo ""
  echo -e "${YELLOW}Environment variables:${NC}"
  echo "  OPENAI_API_KEY - Your OpenAI API key (required for all examples)"
  echo ""
  echo -e "${YELLOW}Example:${NC}"
  echo "  OPENAI_API_KEY=sk-xxxx ./run-examples.sh simple-chat"
}

# Function to check if OpenAI API key is set
check_api_key() {
  if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${RED}Error: OPENAI_API_KEY environment variable is not set.${NC}"
    echo "Please set your OpenAI API key before running the examples:"
    echo "  OPENAI_API_KEY=sk-xxxx ./run-examples.sh $1"
    exit 1
  fi
}

# Function to run an example
run_example() {
  local example=$1
  local class_path=$2
  
  check_api_key "$example"
  
  echo -e "${GREEN}Running $example example...${NC}"
  sbt "examples/runMain $class_path"
}

# List examples
list_examples() {
  echo -e "${CYAN}Available examples:${NC}"
  echo -e "${GREEN}simple-chat${NC}    - Simple chat application using OpenAI"
  echo -e "${GREEN}simple-rag${NC}     - Simple Retrieval Augmented Generation example"
  echo -e "${GREEN}advanced-chat${NC}  - Advanced chat application with memory"
  echo -e "${GREEN}enhanced-rag${NC}   - Enhanced RAG with embeddings and better retrieval"
  echo -e "${GREEN}simple-agent${NC}   - Simple agent example with tool use"
}

# Main execution
if [ $# -eq 0 ]; then
  show_usage
  exit 0
fi

case "$1" in
  "simple-chat")
    run_example "SimpleChat" "zio.langchain.examples.SimpleChat"
    ;;
  "simple-rag")
    run_example "SimpleRAG" "zio.langchain.examples.SimpleRAG"
    ;;
  "advanced-chat")
    run_example "AdvancedChat" "zio.langchain.examples.AdvancedChat"
    ;;
  "enhanced-rag")
    run_example "EnhancedRAG" "zio.langchain.examples.EnhancedRAG"
    ;;
  "simple-agent")
    run_example "SimpleAgent" "zio.langchain.examples.SimpleAgent"
    ;;
  "list")
    list_examples
    ;;
  "help")
    show_usage
    ;;
  *)
    echo -e "${RED}Error: Unknown example '$1'${NC}"
    echo ""
    list_examples
    exit 1
    ;;
esac