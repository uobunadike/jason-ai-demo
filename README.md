# Multi-Persona RAG Assistant

An intelligent AI assistant powered by RAG (Retrieval-Augmented Generation) that supports multiple personas with role-specific knowledge bases.

## Personas

| Persona | Role | Focus Areas |
|---------|------|-------------|
| **Jason** | Inventory Manager | Inventory issues, vendor performance, rush orders, transfers |
| **Claire** | Sales Representative | Customer onboarding, amendments, compliance, commissions |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/query` | Plain text responses |
| POST | `/query-structured` | Structured JSON responses |
| POST | `/compact-query` | Extract numeric KPI values |
| GET | `/suggested-questions/{persona}` | Get 5 suggested questions |
| GET | `/` | Health check |

## Quick Start

### 1. Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
```

### 2. Set Environment Variables
Copy `.env.example` to `.env` and fill in your Azure credentials.

### 3. Run Locally
```bash
python api.py
```

### 4. Test
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Where are my inventory issues?", "persona": "jason"}'
```

## Project Structure

```
├── api.py              # FastAPI application
├── model.py            # Core RAG logic
├── download.py         # Download FAISS index from Azure Blob
├── upload.py           # Upload FAISS index to Azure Blob
├── rebuild_index.py    # Rebuild FAISS index from CSV data
├── test_model.py       # Local testing script
├── data/               # CSV data files
│   ├── jason/          # Inventory data
│   └── claire/         # Onboarding data
├── faiss_index/        # Vector store indexes
│   ├── jason/
│   └── claire/
├── requirements.txt    # Python dependencies
├── Procfile            # Azure deployment config
└── startup.sh          # Azure startup script
```

## Key Features

- **Structured JSON Responses** - Returns formatted data for UI rendering
- **Action Buttons** - Includes actionable buttons (Create Order, Process Transfer)
- **Smart Intent Detection** - Recognizes key questions for each persona
- **Company Name Masking** - Demo-safe responses with masked company names

## Deployment

Deployed on Azure Web App with automatic FAISS index download from Azure Blob Storage.

## License

Proprietary - Internal Use Only

