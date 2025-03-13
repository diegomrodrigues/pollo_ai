# Pollo: AI-Powered Document Generation System

Pollo is an intelligent document generation system that analyzes PDF documents and creates structured, academic-quality content organized by topics and subtopics. It leverages Google's Gemini AI models through LangChain to generate well-structured, detailed drafts suitable for educational or research purposes.

## Key Features

- **PDF Document Analysis**: Extract knowledge from multiple PDF files
- **Intelligent Topic Organization**: Generate a logical structure of topics and subtopics
- **Detailed Draft Generation**: Create comprehensive academic-quality drafts for each subtopic
- **Parallel Processing**: Generate multiple drafts simultaneously for efficiency
- **Multilingual Support**: Generate content in multiple languages (primarily Portuguese)
- **Mathematical Expression Support**: Properly format mathematical notation

## Architecture

Pollo uses a modular architecture based on LangChain's `StateGraph` to orchestrate complex workflows:

```
                   ┌────────────────┐
                   │  PDF Documents │
                   └────────┬───────┘
                            │
                            ▼
           ┌────────────────────────────────┐
           │       Topic Generation         │
           │                                │
           │  1. Extract topics from PDFs   │
           │  2. Generate subtopics         │
           │  3. Consolidate similar topics │
           └──────────────┬─────────────────┘
                          │
                          ▼
           ┌────────────────────────────────┐
           │      Draft Generation          │
           │                                │
           │  1. Generate detailed drafts   │
           │  2. Clean up prompt artifacts  │
           │  3. Generate filenames         │
           └──────────────┬─────────────────┘
                          │
                          ▼
           ┌────────────────────────────────┐
           │      Output Organization       │
           │                                │
           │  1. Create topic directories   │
           │  2. Save drafts with filenames │
           │  3. Organize by topic          │
           └────────────────────────────────┘
```

## Core Components

### 1. Gemini API Integration (`pollo/utils/gemini.py`)

Provides a custom wrapper for Google's Gemini AI models, handling:
- Standard text generation
- File-based content generation (PDF analysis)
- Mock mode for testing without API calls

### 2. YAML Prompt Management (`pollo/utils/prompts.py`)

Loads well-structured prompts from YAML files with fallback options:
```python
prompt = load_chat_prompt_from_yaml(
    "path/to/prompt.yaml", 
    default_system="Fallback system prompt",
    default_user="Fallback user prompt"
)
```

### 3. Topic Generation (`pollo/agents/topics/`)

- `generator.py`: Implements `TopicsGeneratorTool` and LangGraph workflows
- `create_topics.yaml`: Detailed prompt to extract topics from PDFs
- `consolidate_subtopics.yaml`: Prompt for merging similar subtopics

### 4. Draft Generation (`pollo/agents/draft/`)

- `writer.py`: Main draft generation tools:
  - `DraftGeneratorTool`: Creates initial drafts
  - `DraftCleanupTool`: Removes prompt artifacts
  - `FilenameGeneratorTool`: Creates appropriate filenames
- `generate_draft.yaml`: Detailed prompt for academic content creation
- `cleanup_draft.yaml`: Prompt for removing artifacts from generated content
- `generate_filename.yaml`: Prompt for creating standardized filenames

## Workflow

1. **Topic Generation**:
   - PDFs are analyzed by Gemini through multiple perspectives
   - Topics and subtopics are extracted and organized
   - Similar subtopics are consolidated

2. **Draft Generation**:
   - For each subtopic, a detailed draft is generated
   - Drafts include proper mathematical notation and academic structure
   - Multiple drafts can be generated in parallel

3. **Output Organization**:
   - Drafts are cleaned of any prompt artifacts
   - Appropriate filenames are generated
   - Content is saved in a directory structure reflecting the topic hierarchy

## Example Usage

```python
from pollo.agents.draft.writer import create_draft_writer

# Create the draft writer with parallel processing
writer = create_draft_writer(branching_factor=3)

# Generate drafts from PDFs in a directory
result = writer.invoke({
    "directory": "path/to/pdfs",
    "perspectives": ["technical_depth"],
    "json_per_perspective": 3
})

# Result contains paths to generated drafts
print(f"Generated {len(result['drafts'])} drafts in {result['output_directory']}")
```

