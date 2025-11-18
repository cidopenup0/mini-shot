# Plant Disease Detection Frontend

A modern React + Vite + Tailwind CSS frontend for the Plant Disease Detection System.

## Features

- 🖼️ **Image Upload**: Drag-and-drop or browse to upload plant leaf images
- 🔍 **Disease Detection**: AI-powered prediction with confidence scores
- 📊 **Prediction History**: View all past predictions
- 📋 **Disease Classes**: Browse 38 supported plant diseases
- 💚 **Health Monitoring**: Real-time backend and model status
- 📱 **Responsive Design**: Works on desktop, tablet, and mobile
- ⚡ **Fast & Modern**: Built with Vite for lightning-fast development

## Tech Stack

- **React 18** - UI library
- **Vite** - Build tool and dev server
- **Tailwind CSS** - Utility-first styling
- **React Router** - Client-side routing
- **Axios** - HTTP client for API calls

## Prerequisites

- Node.js 16+ and npm/yarn
- FastAPI backend running on `http://localhost:8000`

## Installation

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The app will open at `http://localhost:3000`

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build

## Project Structure

```
frontend/
├── src/
│   ├── components/       # Reusable UI components
│   │   ├── ClassesList.jsx
│   │   ├── ErrorBox.jsx
│   │   ├── HistoryTable.jsx
│   │   ├── ImageUpload.jsx
│   │   ├── Loader.jsx
│   │   ├── PreviewCard.jsx
│   │   └── PredictionCard.jsx
│   ├── pages/           # Page components
│   │   ├── Home.jsx
│   │   ├── History.jsx
│   │   ├── Classes.jsx
│   │   └── Health.jsx
│   ├── services/        # API client
│   │   └── api.js
│   ├── utils/           # Helper functions
│   │   └── helpers.js
│   ├── App.jsx          # Main app component with routing
│   ├── main.jsx         # Entry point
│   └── index.css        # Global styles
├── index.html           # HTML template
├── package.json         # Dependencies
├── vite.config.js       # Vite configuration
├── tailwind.config.js   # Tailwind configuration
└── postcss.config.js    # PostCSS configuration
```

## API Integration

The frontend connects to these FastAPI endpoints:

- `POST /predict` - Upload image and get prediction
- `GET /history` - Fetch prediction history
- `GET /classes` - Get list of disease classes
- `GET /health` - Check backend health status

## Features in Detail

### Home Page
- Upload plant leaf images (JPEG/PNG, max 10MB)
- Drag-and-drop support
- Image preview
- Real-time prediction with confidence score
- Visual confidence indicator

### History Page
- View all past predictions
- Sortable table with timestamps
- Responsive card view for mobile

### Classes Page
- Browse 38 supported disease classes
- Grouped by plant type (Tomato, Potato, Pepper)
- Search functionality

### Health Page
- Backend connection status
- Model load status
- Auto-refresh every 30 seconds

## Customization

### Colors
Edit `tailwind.config.js` to change the primary color scheme.

### API URL
Update `API_BASE_URL` in `src/services/api.js` if your backend runs on a different port.

## Building for Production

```bash
npm run build
```

This creates optimized files in the `dist/` folder ready for deployment.

## Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## License

MIT
