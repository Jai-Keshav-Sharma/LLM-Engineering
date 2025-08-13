# 🏢 Week 5: Retrieval-Augmented Generation (RAG) Systems

![RAG Systems](https://img.shields.io/badge/RAG-Systems-blue?style=for-the-badge)
![Vector Databases](https://img.shields.io/badge/Vector-Databases-green?style=for-the-badge)
![LangChain](https://img.shields.io/badge/LangChain-Framework-orange?style=for-the-badge)
![Gradio](https://img.shields.io/badge/Gradio-UI-purple?style=for-the-badge)

## 📖 Table of Contents
- [🎯 Week Overview](#-week-overview)
- [📊 Learning Architecture](#-learning-architecture)
- [📚 Lab Breakdown](#-lab-breakdown)
  - [Lab 1: RAG from Scratch](#lab-1-rag-from-scratch-brute-forcing-rag)
  - [Lab 2: Document Chunking & Text Search](#lab-2-document-chunking--text-search)
  - [Lab 3: Vector Embeddings & Visualization](#lab-3-vector-embeddings--visualization)
  - [Lab 4: Expert Knowledge Worker](#lab-4-expert-knowledge-worker)
  - [Lab 4.5: FAISS vs ChromaDB Comparison](#lab-45-faiss-vs-chromadb-comparison)
  - [Lab 5: RAG Debugging & Optimization](#lab-5-rag-debugging--optimization)
- [🏗️ Knowledge Base Structure](#-knowledge-base-structure)
- [🔧 RAG Pipeline Architecture](#-rag-pipeline-architecture)
- [🎨 Vector Visualization](#-vector-visualization)
- [💡 Key Insights](#-key-insights)
- [🚀 Deployment](#-deployment)

## 🎯 Week Overview

Week 5 focuses on building **production-ready RAG (Retrieval-Augmented Generation) systems** for enterprise knowledge management. We start from basic text matching and gradually work our way up to sophisticated vector-based retrieval systems. Along the journey, we build an intelligent knowledge worker for Insurellm (an InsurTech company) and then dive deep into debugging when things don't work as expected.

```mermaid
flowchart TD
    A[Week 5: RAG Systems] --> B[Lab 1: Basic RAG]
    A --> C[Lab 2: Chunking Strategy]
    A --> D[Lab 3: Vector Embeddings]
    A --> E[Lab 4: Production RAG]
    A --> F[Lab 4.5: Vector DB Comparison]
    A --> G[Lab 5: Debugging & Fixes]
    
    B --> H[Manual Text Matching]
    B --> I[Direct OpenAI Integration]
    B --> J[Simple Knowledge Retrieval]
    
    C --> K[Document Loading]
    C --> L[Text Splitting]
    C --> M[Metadata Management]
    
    D --> N[OpenAI Embeddings]
    D --> O[ChromaDB Storage]
    D --> P[t-SNE Visualization]
    
    E --> Q[Conversational RAG]
    E --> R[React Agent]
    E --> S[Gradio Interface]
    
    F --> T[FAISS Implementation]
    F --> U[Performance Comparison]
    F --> V[3D Visualization]
    
    G --> W[LangChain Callbacks]
    G --> X[RAG Failure Analysis]
    G --> Y[Optimization Strategies]
    
    style A fill:#e1f5fe
    style B fill:#fff3e0
    style C fill:#e8f5e8
    style D fill:#f3e5f5
    style E fill:#fce4ec
    style F fill:#e8f5e8
    style G fill:#ffebee
```

## 📊 Learning Architecture

```mermaid
gantt
    title Week 5 Learning Journey
    dateFormat X
    axisFormat %d
    
    section Lab 1: Basic RAG
    Manual Implementation    :1, 3
    Text Matching           :2, 4
    Simple Retrieval        :3, 5
    
    section Lab 2: Chunking
    Document Loading        :4, 6
    Text Splitting          :5, 7
    Metadata Enrichment     :6, 8
    
    section Lab 3: Embeddings
    Vector Generation       :7, 9
    ChromaDB Integration    :8, 10
    Visualization          :9, 11
    
    section Lab 4: Production
    Conversational Chain    :10, 12
    React Agent            :11, 13
    UI Development         :12, 14
    
    section Lab 4.5: Vector DB
    FAISS Implementation   :13, 15
    Performance Testing    :14, 16
    3D Visualization      :15, 17
    
    section Lab 5: Debug & Fix
    RAG Failure Analysis   :16, 18
    Callback Investigation :17, 19
    Performance Tuning     :18, 20
```

## 📚 Lab Breakdown

### Lab 1: RAG from Scratch (Brute Forcing RAG)
**[📓 1_lab.ipynb](1_lab.ipynb)**

This lab demonstrates the fundamental concept of RAG by implementing a basic version from scratch using manual text matching and direct OpenAI API calls.

#### 🔧 Core Components

| Component | Purpose | Implementation |
|-----------|---------|----------------|
| **Knowledge Loading** | Extract documents from files | Manual file reading with glob |
| **Simple Matching** | Basic text search | String matching algorithms |
| **Context Assembly** | Combine relevant information | Dictionary-based storage |
| **LLM Integration** | Generate responses | Direct OpenAI API calls |

#### 💡 Key Learning Points

- **Manual Implementation**: Understanding RAG principles without frameworks
- **File System Navigation**: Working with directory structures and file paths
- **Context Management**: Building simple knowledge dictionaries
- **Basic Retrieval**: Implementing search without vector similarity

```mermaid
sequenceDiagram
    participant User
    participant App
    participant Files
    participant OpenAI
    
    User->>App: Ask Question
    App->>Files: Search Knowledge Base
    Files-->>App: Return Matching Text
    App->>OpenAI: Send Context + Query
    OpenAI-->>App: Generate Response
    App-->>User: Return Answer
```

---

### Lab 2: Document Chunking & Text Search
**[📓 2_lab.ipynb](2_lab.ipynb)**

Advanced document processing using LangChain for systematic chunking and metadata management.

#### 🔧 LangChain Integration

| Feature | Technology | Purpose |
|---------|------------|---------|
| **Document Loaders** | DirectoryLoader, TextLoader | Systematic file ingestion |
| **Text Splitting** | CharacterTextSplitter | Optimal chunk creation |
| **Metadata Management** | Custom doc_type tags | Document categorization |
| **Batch Processing** | Automated pipeline | Handle large document sets |

#### 📋 Chunking Strategy

```mermaid
graph TD
    A[Raw Documents] --> B[DirectoryLoader]
    B --> C[Add Metadata]
    C --> D[CharacterTextSplitter]
    D --> E[Chunks with Metadata]
    
    F[Chunk Parameters]
    F --> G[Size: 1000 chars]
    F --> H[Overlap: 200 chars]
    F --> I[Preserve Context]
    
    G --> D
    H --> D
    I --> D
    
    style A fill:#e1f5fe
    style E fill:#e8f5e8
    style F fill:#fff3e0
```

#### 🎯 Advanced Features

- **Metadata Enrichment**: Automatic document type classification
- **Overlap Strategy**: Preventing context loss at chunk boundaries  
- **Unicode Handling**: Proper encoding for special characters
- **Scalable Processing**: Efficient handling of large document collections

---

### Lab 3: Vector Embeddings & Visualization
**[📓 3_lab.ipynb](3_lab.ipynb)**

Implementation of semantic search using OpenAI embeddings and ChromaDB vector storage with advanced visualization techniques.

#### 🧠 Vector Processing Pipeline

```mermaid
flowchart LR
    A[Text Chunks] --> B[OpenAI Embeddings]
    B --> C[1536-D Vectors]
    C --> D[ChromaDB Storage]
    D --> E[Similarity Search]
    
    F[Query Text] --> G[Query Embedding]
    G --> H[Vector Similarity]
    H --> I[Relevant Chunks]
    
    C --> J[t-SNE Reduction]
    J --> K[2D Visualization]
    K --> L[Interactive Plots]
    
    style B fill:#fff3e0
    style D fill:#e8f5e8
    style J fill:#f3e5f5
```

#### 📊 Technical Specifications

| Component | Specification | Purpose |
|-----------|--------------|---------|
| **Embedding Model** | OpenAI text-embedding-ada-002 | 1536-dimensional vectors |
| **Vector Database** | ChromaDB | Persistent similarity search |
| **Dimensionality Reduction** | t-SNE | 2D visualization |
| **Visualization** | Plotly | Interactive scatter plots |

#### 🎨 Visualization Features

- **Color Coding**: Different colors for document types (employees, products, contracts, company)
- **Interactive Hover**: Text previews and metadata on hover
- **Cluster Analysis**: Visual representation of semantic similarity
- **Performance Metrics**: Vector dimensionality and collection size

---

### Lab 4: Expert Knowledge Worker
**[📓 4_lab.ipynb](4_lab.ipynb)**

Production-ready RAG system with conversational memory and React agents, designed as an expert knowledge worker for Insurellm.

#### 🏢 Business Context

**Insurellm** - Insurance Technology Company
- Founded by Avery Lancaster in 2015
- 200+ employees across 12 US offices
- Product portfolio: Markellm, Carllm, Homellm, Rellm
- Focus: Disrupting traditional insurance with innovative tech

#### 🤖 System Architecture

```mermaid
flowchart TD
    A[Gradio Chat Interface] --> B[React Agent]
    B --> C[Conversation Memory]
    B --> D[Vector Store Retriever]
    D --> E[ChromaDB]
    E --> F[Similarity Search]
    
    F --> G[Company Info]
    F --> H[Employee Profiles]
    F --> I[Product Details]
    F --> J[Contract Data]
    
    B --> K[GPT-4o-mini]
    K --> L[Response Generation]
    L --> A
    
    M[User Query] --> A
    A --> N[Response Display]
    
    style A fill:#e1f5fe
    style B fill:#fff3e0
    style D fill:#e8f5e8
    style K fill:#f3e5f5
    style E fill:#e8f5e8
```

#### 🔧 Implementation Details

**Conversational RAG Chain:**
- **LLM**: GPT-4o-mini for cost-effective responses
- **Memory**: ConversationBufferMemory for context retention
- **Retriever**: Vector store abstraction over ChromaDB
- **Chain**: ConversationalRetrievalChain for seamless integration

**React Agent:**
- **Tools**: Retriever tool for knowledge base access
- **Memory**: MemorySaver for persistent conversations
- **Planning**: Automatic tool selection and reasoning
- **Execution**: Multi-step problem solving

#### 💬 User Interface Features

```mermaid
graph LR
    A[Chat Interface] --> B[Question Examples]
    A --> C[Retry Functionality]
    A --> D[Conversation History]
    A --> E[Clear/Undo Options]
    
    B --> F["Who is Avery Lancaster?"]
    B --> G["What products does Insurellm offer?"]
    B --> H["Tell me about our contracts"]
    
    style A fill:#e1f5fe
    style B fill:#e8f5e8
```

---

### Lab 4.5: FAISS vs ChromaDB Comparison  
**[📓 4.5_lab.ipynb](4.5_lab.ipynb)**

Ever wondered about the differences between vector databases? This lab takes the same RAG system from Lab 4 and rebuilds it using FAISS instead of ChromaDB, allowing us to compare these two popular vector storage solutions side by side.

#### 🔄 The Experiment

We implement the exact same RAG pipeline using FAISS to see how it compares with ChromaDB in terms of:
- **Setup complexity**
- **Performance characteristics** 
- **Visualization capabilities**
- **Integration with LangChain**

#### 📊 ChromaDB vs FAISS: Detailed Comparison

```mermaid
graph TB
    subgraph "ChromaDB Characteristics"
        A1[Persistent Storage]
        A2[Built-in Metadata Support]
        A3[CRUD Operations]
        A4[Collection Management]
        A5[Easy LangChain Integration]
    end
    
    subgraph "FAISS Characteristics"
        B1[In-Memory by Default]
        B2[Blazing Fast Search]
        B3[Advanced Index Types]
        B4[Memory Efficient]
        B5[Facebook Research Optimized]
    end
    
    subgraph "Common Features"
        C1[Vector Similarity Search]
        C2[Embedding Storage]
        C3[Scalable Architecture]
        C4[Python APIs]
    end
    
    style A1 fill:#e1f5fe
    style B1 fill:#fff3e0
    style C1 fill:#e8f5e8
```

#### ⚡ Performance & Architecture Comparison

| Aspect | ChromaDB | FAISS | Winner |
|--------|----------|-------|---------|
| **Setup Complexity** | Simple, plug-and-play | Requires more configuration | 🟦 ChromaDB |
| **Search Speed** | Good for most use cases | Extremely fast | 🟨 FAISS |
| **Memory Usage** | Higher overhead | Memory optimized | 🟨 FAISS |
| **Persistence** | Built-in disk storage | Manual save/load | 🟦 ChromaDB |
| **Metadata Handling** | Native support | Requires workarounds | 🟦 ChromaDB |
| **Scalability** | Good for medium datasets | Excellent for large datasets | 🟨 FAISS |
| **Index Types** | Standard similarity | Multiple algorithms (IVF, HNSW, etc.) | 🟨 FAISS |
| **Production Ready** | Yes, with clustering | Yes, battle-tested | 🤝 Tie |

#### 🎨 Visualization Enhancements

One cool feature in this lab is the **3D visualization** of our vector space:

```python
# 3D t-SNE visualization with FAISS
tsne = TSNE(n_components=3, random_state=42)
reduced_vectors = tsne.fit_transform(vectors)

fig = go.Figure(data=[go.Scatter3d(
    x=reduced_vectors[:, 0],
    y=reduced_vectors[:, 1], 
    z=reduced_vectors[:, 2],
    mode='markers',
    marker=dict(size=5, color=colors, opacity=0.8)
)])
```

This gives us a more immersive view of how our document chunks cluster in the embedding space!

#### 🔧 Implementation Differences

**ChromaDB Approach:**
```python
# Simple and straightforward
vectorstore = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings, 
    persist_directory=db_name
)
```

**FAISS Approach:**
```python
# More direct, in-memory focused
vectorstore = FAISS.from_documents(
    documents=chunks, 
    embedding=embeddings
)

# Access internal structures for analysis
total_vectors = vectorstore.index.ntotal
dimensions = vectorstore.index.d
```

#### 🎯 Key Insights from the Comparison

**When to Choose ChromaDB:**
- You need persistent storage out of the box
- Metadata filtering is important for your use case
- You want simple setup and maintenance
- Your dataset is small to medium sized (< 1M vectors)
- You're building a prototype or MVP

**When to Choose FAISS:**
- You need maximum search performance
- Working with large datasets (1M+ vectors)
- Memory efficiency is critical
- You need specific index types (IVF, HNSW, etc.)
- You're building high-throughput production systems

**Real-World Usage Patterns:**
- **Startups/Prototypes**: ChromaDB for rapid development
- **Large Scale Production**: FAISS for performance-critical applications
- **Hybrid Approach**: ChromaDB for development, FAISS for production
- **Enterprise**: Often both, depending on specific use cases

#### 💡 Practical Takeaways

- **Both work great with LangChain** - the choice often comes down to specific requirements
- **FAISS requires more manual work** for persistence and metadata management
- **ChromaDB is more beginner-friendly** but FAISS offers more control
- **Performance differences matter most at scale** - for small datasets, either works fine
- **Consider your team's expertise** - FAISS has a steeper learning curve

This comparison helps you make informed decisions about vector database choice based on your specific needs rather than just following trends!

---

### Lab 5: RAG Debugging & Optimization
**[📓 5_lab.ipynb](5_lab.ipynb)**

Sometimes RAG systems don't work as expected, even when everything looks perfect on paper. This lab explores what happens when our knowledge worker fails to find information we *know* exists in the knowledge base.

#### 🐛 The Problem

Ever had that frustrating moment when you ask your RAG system about something you *know* is in your documents, but it just says "I don't know"? That's exactly what happened here:

**Question**: "Who received the prestigious IIOTY award in 2023?"  
**RAG Response**: "I don't know"  
**Reality**: Maxine Thompson's employee record clearly mentions she received the IIOTY 2023 award

#### 🔍 Detective Work with LangChain Callbacks

Instead of just accepting the failure, we dig deeper using LangChain's callback system to see exactly what's happening under the hood:

```mermaid
flowchart LR
    A[User Query] --> B[Vector Similarity Search]
    B --> C[Retrieved Chunks]
    C --> D[LLM Context]
    D --> E[Generated Response]
    
    F[StdOutCallbackHandler] --> G[Debug Output]
    G --> H[See Retrieved Chunks]
    H --> I[Identify Missing Info]
    
    B --> F
    C --> F
    D --> F
    
    style F fill:#ffebee
    style G fill:#fff3e0
    style I fill:#e8f5e8
```

#### 🛠️ Debugging Techniques

| Approach | Tool | What It Reveals |
|----------|------|----------------|
| **Callback Inspection** | `StdOutCallbackHandler` | Which chunks were actually retrieved |
| **Chunk Analysis** | Manual verification | Whether relevant info exists in chunks |
| **Similarity Testing** | Vector distance checks | Why certain chunks weren't retrieved |
| **Context Limits** | Token counting | If context window is the bottleneck |

#### 💡 Mitigation Strategies

When RAG fails, here are the strategies we explore:

**1. Adjust Retrieval Parameters**
```python
# Increase the number of chunks retrieved
retriever = vectorstore.as_retriever(search_kwargs={'k': 25})  # Instead of default k=4
```

**2. Examine Chunking Strategy**
- Are chunks too small/large?
- Is overlap sufficient to preserve context?
- Does the chunk boundary split important information?

**3. Debug with Callbacks**
```python
# See exactly what chunks are being used
conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm, 
    retriever=retriever, 
    memory=memory, 
    callbacks=[StdOutCallbackHandler()]
)
```

**4. Consider Alternative Approaches**
- Semantic chunking instead of character-based
- Different embedding models
- Hybrid search (keyword + vector)
- Query rewriting/expansion

#### 🎯 Real-World Insights

This lab shows that building RAG systems isn't just about the happy path - it's about understanding when and why they fail, then systematically debugging and improving them. Sometimes the issue is:

- **Chunk boundaries** splitting related information
- **Vector similarity** not capturing the right semantic relationships  
- **Retrieval parameters** being too restrictive
- **Query phrasing** not matching document language

#### 🔧 Practical Takeaways

- **Always debug with callbacks** when RAG fails unexpectedly
- **Test edge cases** beyond the obvious questions
- **Monitor chunk quality** and retrieval patterns
- **Iterate on parameters** rather than assuming defaults work best
- **Remember**: RAG is as much about debugging as it is about building

## 🏗️ Knowledge Base Structure

Our enterprise knowledge base is organized into four main categories:

```
knowledge-base/
├── 📁 company/
│   ├── about.md          # Company overview and history
│   ├── careers.md        # Career opportunities and culture
│   └── overview.md       # Business model and strategy
├── 📁 employees/
│   ├── Avery Lancaster.md    # Founder profile
│   ├── Alex Chen.md          # Employee profiles
│   ├── Emily Carter.md       # Team information
│   └── [12 employee files]  # Complete team directory
├── 📁 products/
│   ├── Markellm.md      # Insurance marketplace
│   ├── Carllm.md        # Auto insurance AI
│   ├── Homellm.md       # Home insurance AI
│   └── Rellm.md         # Reinsurance AI
└── 📁 contracts/
    ├── Contract with Apex Reinsurance.md
    ├── Contract with Belvedere Insurance.md
    └── [12 contract files]
```

### 📊 Content Statistics

| Category | Files | Avg. Size | Content Type |
|----------|-------|-----------|--------------|
| **Company** | 3 | 1.2KB | Strategic info, culture |
| **Employees** | 12 | 0.8KB | Bios, roles, expertise |
| **Products** | 4 | 1.5KB | Features, use cases |
| **Contracts** | 12 | 2.1KB | Terms, partnerships |

## 🔧 RAG Pipeline Architecture

### Complete Information Flow

```mermaid
flowchart TD
    A[User Query] --> B[Query Processing]
    B --> C[Vector Embedding]
    C --> D[Similarity Search]
    D --> E[Chunk Retrieval]
    E --> F[Context Assembly]
    F --> G[Prompt Construction]
    G --> H[LLM Generation]
    H --> I[Response Formatting]
    I --> J[User Response]
    
    K[Knowledge Base] --> L[Document Loading]
    L --> M[Text Chunking]
    M --> N[Metadata Tagging]
    N --> O[Vector Generation]
    O --> P[ChromaDB Storage]
    P --> D
    
    style A fill:#e1f5fe
    style H fill:#fff3e0
    style P fill:#e8f5e8
    style J fill:#f3e5f5
```

### Technical Stack

```mermaid
graph TB
    subgraph "Frontend"
        A[Gradio Interface]
    end
    
    subgraph "Framework"
        B[LangChain]
        C[LangGraph Agents]
    end
    
    subgraph "Vector Store"
        D[ChromaDB]
        E[OpenAI Embeddings]
    end
    
    subgraph "LLM"
        F[GPT-4o-mini]
        G[OpenAI API]
    end
    
    subgraph "Data"
        H[Markdown Files]
        I[Metadata]
    end
    
    A --> B
    B --> C
    C --> D
    D --> E
    B --> F
    F --> G
    H --> I
    I --> D
    
    style A fill:#e1f5fe
    style B fill:#fff3e0
    style D fill:#e8f5e8
    style F fill:#f3e5f5
```

## 🎨 Vector Visualization

### Embedding Space Analysis

The t-SNE visualization reveals semantic clustering of document chunks:

```mermaid
graph LR
    A[1536D OpenAI Embeddings] --> B[t-SNE Reduction]
    B --> C[2D Scatter Plot]
    
    D[Color Coding]
    D --> E[🔵 Products: Blue]
    D --> F[🟢 Employees: Green]
    D --> G[🔴 Contracts: Red]
    D --> H[🟠 Company: Orange]
    
    C --> I[Interactive Visualization]
    I --> J[Hover Information]
    I --> K[Cluster Analysis]
    I --> L[Semantic Relationships]
    
    style A fill:#fff3e0
    style C fill:#e8f5e8
    style I fill:#f3e5f5
```

### Insights from Visualization

- **Semantic Clustering**: Similar content types cluster together
- **Cross-Category Relationships**: Some overlap between employee and product information
- **Quality Assessment**: Well-distributed embeddings indicate good content diversity
- **Dimensionality**: 1536-dimensional vectors effectively capture semantic meaning

## 💡 Key Insights

### 🎯 Technical Achievements

1. **Progressive Complexity**: From manual implementation to production-ready systems
2. **Framework Mastery**: Deep understanding of LangChain ecosystem
3. **Vector Operations**: Hands-on experience with embeddings and similarity search
4. **Agent Architecture**: Implementation of reasoning agents with tool use
5. **UI Development**: Professional chat interfaces with Gradio

### 🏢 Business Value

1. **Cost Efficiency**: GPT-4o-mini provides excellent performance at lower cost
2. **Accuracy**: RAG ensures responses are grounded in company knowledge
3. **Scalability**: Vector database supports growing knowledge bases
4. **User Experience**: Intuitive chat interface for non-technical users
5. **Memory**: Conversational context maintains engagement

### 🔬 Learning Outcomes

```mermaid
mindmap
  root((RAG Mastery))
    Technical Skills
      Vector Embeddings
      ChromaDB Operations
      LangChain Framework
      React Agents
    Business Understanding
      Insurance Domain
      Knowledge Management
      User Experience
      Cost Optimization
    Implementation
      Document Processing
      Chunking Strategies
      Similarity Search
      Conversational AI
    Visualization
      t-SNE Reduction
      Interactive Plots
      Cluster Analysis
      Performance Metrics
```

## 🚀 Deployment

### Quick Start Guide

1. **Environment Setup**:
   ```bash
   pip install langchain langchain-openai langchain-chroma
   pip install gradio plotly scikit-learn
   pip install chromadb openai python-dotenv
   ```

2. **Environment Variables**:
   ```bash
   OPENAI_API_KEY=your_openai_api_key
   ```

3. **Run the Knowledge Worker**:
   ```python
   # From Lab 4
   demo.launch(inbrowser=True)
   ```

### Production Considerations

- **Scaling**: ChromaDB supports distributed deployments
- **Security**: Implement authentication for sensitive knowledge
- **Monitoring**: Add logging and performance metrics
- **Updates**: Automated pipeline for knowledge base updates
- **Backup**: Regular vector database backups

---

## 🔗 Navigation Links

- [📓 Lab 1: Basic RAG Implementation](1_lab.ipynb)
- [📓 Lab 2: Document Chunking & Search](2_lab.ipynb)
- [📓 Lab 3: Vector Embeddings & Visualization](3_lab.ipynb)
- [📓 Lab 4: Expert Knowledge Worker](4_lab.ipynb)
- [📓 Lab 4.5: FAISS vs ChromaDB Comparison](4.5_lab.ipynb)
- [📓 Lab 5: RAG Debugging & Optimization](5_lab.ipynb)
- [📁 Knowledge Base](knowledge-base/)
- [🏠 Repository Home](../README.md)
- [📊 Previous Week](../4_week/README.md)

---

*Built with ❤️ using LangChain, ChromaDB, OpenAI, and Gradio. Empowering Insurellm with intelligent knowledge management.*
