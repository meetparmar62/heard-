# Heart Disease Prediction - Data Flow Diagram

## System Architecture Overview

```mermaid
flowchart TB
    subgraph User_Interface["🖥️ User Interface Layer (Streamlit)"]
        A[User Input Form<br/>Medical Parameters] --> B[Data Validation<br/>& Processing]
        B --> C[Display Results<br/>Prediction & Analytics]
    end
    
    subgraph API_Layer["⚡ API Layer (FastAPI)"]
        D[REST API Endpoint<br/>POST /predict/] --> E[Request Validation<br/>& Serialization]
        E --> F[Response Handler<br/>JSON Format]
    end
    
    subgraph ML_Layer["🤖 Machine Learning Layer"]
        G[Input Preprocessing<br/>Scaling & Encoding] --> H[Random Forest<br/>Classifier Model]
        H --> I[Prediction<br/>Probability Score]
    end
    
    subgraph Storage["💾 Storage Layer"]
        J[Model File<br/>random_forest_model.pkl]
        K[Preprocessor<br/>preprocessor.pkl]
    end
    
    C -->|1. User submits form data| D
    B -->|2. Validated input data| A
    D -->|3. JSON payload| G
    G -->|4. Processed features| H
    H -->|5. Prediction result| I
    I -->|6. Probability scores| F
    F -->|7. Prediction response| C
    J -.->|Load model| H
    K -.->|Load preprocessor| G
    
    style User_Interface fill:#e3f2fd,stroke:#1976d2,stroke-width:3px
    style API_Layer fill:#f3e5f5,stroke:#7b1fa2,stroke-width:3px
    style ML_Layer fill:#e8f5e9,stroke:#388e3c,stroke-width:3px
    style Storage fill:#fff3e0,stroke:#f57c00,stroke-width:3px
    style A fill:#bbdefb,stroke:#1976d2,stroke-width:2px
    style B fill:#bbdefb,stroke:#1976d2,stroke-width:2px
    style C fill:#bbdefb,stroke:#1976d2,stroke-width:2px
    style D fill:#e1bee7,stroke:#7b1fa2,stroke-width:2px
    style E fill:#e1bee7,stroke:#7b1fa2,stroke-width:2px
    style F fill:#e1bee7,stroke:#7b1fa2,stroke-width:2px
    style G fill:#c8e6c9,stroke:#388e3c,stroke-width:2px
    style H fill:#c8e6c9,stroke:#388e3c,stroke-width:2px
    style I fill:#c8e6c9,stroke:#388e3c,stroke-width:2px
    style J fill:#ffe0b2,stroke:#f57c00,stroke-width:2px
    style K fill:#ffe0b2,stroke:#f57c00,stroke-width:2px
```

## Detailed Request-Response Flow

```mermaid
sequenceDiagram
    participant User
    participant UI as Streamlit UI
    participant API as FastAPI
    participant ML as ML Model
    participant DB as Model Storage
    
    User->>UI: Enter medical parameters
    Note over User,UI: Age, BP, Cholesterol, etc.
    
    UI->>UI: Validate input data
    Note right of UI: Check ranges & types
    
    UI->>API: POST /predict/<br/>JSON payload
    Note over UI,API: HTTP Request
    
    API->>API: Deserialize JSON
    Note right of API: Convert to DataFrame
    
    API->>ML: Send features
    Note over API,ML: Preprocessed input
    
    ML->>DB: Load model & preprocessor
    Note over ML,DB: Pickle files
    
    DB-->>ML: Return loaded models
    
    ML->>ML: Apply preprocessing
    Note right of ML: Scaling & encoding
    
    ML->>ML: Random Forest prediction
    Note right of ML: Predict + probabilities
    
    ML-->>API: Return prediction
    Note over ML,API: {prediction: 0/1,<br/>probability: float}
    
    API-->>UI: JSON response
    Note over API,UI: Status 200 OK
    
    UI->>UI: Process results
    Note right of UI: Risk assessment
    
    UI-->>User: Display prediction
    Note over UI,User: Risk level + graphs
    
    Note over User,User: 💡 Consult doctor if high risk
```

## Component Interaction Diagram

```mermaid
graph LR
    subgraph Frontend["Frontend Components"]
        A[Input Form] --> B[Validation Logic]
        B --> C[API Client]
        C --> D[Result Display]
        D --> E[Analytics Dashboard]
    end
    
    subgraph Backend["Backend Services"]
        F[API Router] --> G[Request Handler]
        G --> H[Data Processor]
        H --> I[Prediction Service]
    end
    
    subgraph Models["ML Infrastructure"]
        J[Feature Preprocessor]
        K[Random Forest Model]
        L[Prediction Pipeline]
    end
    
    E -.->|Triggers prediction| F
    I -->|Uses| J
    I -->|Loads| K
    J -->|Feeds into| K
    K -->|Generates| L
    L -->|Returns| I
    I -->|Sends result| G
    G -->|Response| F
    F -->|JSON data| C
    
    style Frontend fill:#e3f2fd,stroke:#1976d2,stroke-width:3px
    style Backend fill:#f3e5f5,stroke:#7b1fa2,stroke-width:3px
    style Models fill:#e8f5e9,stroke:#388e3c,stroke-width:3px
```

## Data Transformation Pipeline

```mermaid
flowchart LR
    A[Raw User Input<br/>JSON Object] --> B[Data Parsing<br/>Dictionary]
    B --> C[DataFrame Creation<br/>Pandas DataFrame]
    C --> D[Feature Scaling<br/>StandardScaler]
    D --> E[Categorical Encoding<br/>OneHotEncoder]
    E --> F[Final Feature Vector<br/>Numeric Array]
    F --> G[Model Inference<br/>Random Forest]
    G --> H[Prediction Output<br/>Binary Class 0/1]
    H --> I[Probability Scores<br/>Confidence %]
    I --> J[Result Formatting<br/>JSON Response]
    
    style A fill:#bbdefb,stroke:#1976d2,stroke-width:2px
    style B fill:#bbdefb,stroke:#1976d2,stroke-width:2px
    style C fill:#bbdefb,stroke:#1976d2,stroke-width:2px
    style D fill:#c8e6c9,stroke:#388e3c,stroke-width:2px
    style E fill:#c8e6c9,stroke:#388e3c,stroke-width:2px
    style F fill:#c8e6c9,stroke:#388e3c,stroke-width:2px
    style G fill:#e1bee7,stroke:#7b1fa2,stroke-width:2px
    style H fill:#e1bee7,stroke:#7b1fa2,stroke-width:2px
    style I fill:#ffe0b2,stroke:#f57c00,stroke-width:2px
    style J fill:#bbdefb,stroke:#1976d2,stroke-width:2px
```

## Complete System Flow with Analytics

```mermaid
flowchart TB
    Start([👤 User Opens App]) --> Input[📝 Fill Medical Details Form]
    Input --> Validate{✅ Valid Data?}
    
    Validate -->|No| Error[❌ Show Error Message]
    Error --> Input
    
    Validate -->|Yes| Process[⚙️ Process & Validate]
    Process --> Send[📤 Send to API]
    
    Send --> Preprocess[🔄 Preprocess Data<br/>Scaling + Encoding]
    Preprocess --> Predict[🤖 ML Model Prediction<br/>Random Forest]
    
    Predict --> Result{Prediction?}
    Result -->|High Risk| Warning[⚠️ High Risk Alert]
    Result -->|Low Risk| Success[✅ Low Risk Confirmation]
    
    Warning --> Display[📊 Display Results + Graphs]
    Success --> Display
    
    Display --> Analytics[📈 View Analytics Dashboard]
    Analytics --> Charts{Select Visualization}
    
    Charts -->|Option 1| Dist[Distribution Analysis]
    Charts -->|Option 2| Corr[Correlation Heatmap]
    Charts -->|Option 3| Feature[Feature Importance]
    Charts -->|Option 4| Risk[Risk Factor Analysis]
    Charts -->|Option 5| Demo[Age & Gender Distribution]
    Charts -->|Option 6| Medical[Medical Parameters]
    
    Dist --> End([End Session])
    Corr --> End
    Feature --> End
    Risk --> End
    Demo --> End
    Medical --> End
    
    style Start fill:#bbdefb,stroke:#1976d2,stroke-width:3px
    style Input fill:#bbdefb,stroke:#1976d2,stroke-width:2px
    style Validate fill:#ffe0b2,stroke:#f57c00,stroke-width:2px
    style Error fill:#ffcdd2,stroke:#d32f2f,stroke-width:2px
    style Process fill:#c8e6c9,stroke:#388e3c,stroke-width:2px
    style Send fill:#c8e6c9,stroke:#388e3c,stroke-width:2px
    style Preprocess fill:#e1bee7,stroke:#7b1fa2,stroke-width:2px
    style Predict fill:#e1bee7,stroke:#7b1fa2,stroke-width:2px
    style Result fill:#ffe0b2,stroke:#f57c00,stroke-width:2px
    style Warning fill:#ffcdd2,stroke:#d32f2f,stroke-width:2px
    style Success fill:#c8e6c9,stroke:#388e3c,stroke-width:2px
    style Display fill:#bbdefb,stroke:#1976d2,stroke-width:2px
    style Analytics fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style End fill:#cfd8dc,stroke:#546e7a,stroke-width:3px
```

---

## Usage Instructions

### For Presentations:

1. **Copy any diagram** and paste it in:
   - GitHub README.md
   - Notion documents
   - Markdown editors with Mermaid support
   - VS Code with Mermaid extension

2. **Render online** at:
   - [Mermaid Live Editor](https://mermaid.live/)
   - [GitLab/GitHub](native support)

3. **Export as image**:
   - Use Mermaid CLI: `mmdc -i diagram.mmd -o diagram.png`
   - Screenshot from live editor
   - Browser DevTools → Capture node screenshot

### Color Scheme:
- 🔵 **Light Blue** (#e3f2fd, #bbdefb) - User Interface
- 🟣 **Light Purple** (#f3e5f5, #e1bee7) - API Layer
- 🟢 **Light Green** (#e8f5e9, #c8e6c9) - ML Layer
- 🟠 **Light Orange** (#fff3e0, #ffe0b2) - Storage

### Perfect for College Presentation! 🎓
