# Frontend - Plant Disease Detection System

## Overview
This frontend is built with React, Vite, and Tailwind CSS. It provides a modern, responsive UI for uploading plant leaf images, viewing predictions, submitting feedback, and exploring disease classes.

## Features
- Upload leaf images for disease prediction
- View prediction results and confidence
- Submit feedback (correct/incorrect, corrected class)
- View prediction history and feedback status
- Explore all supported disease classes
- Check backend/model health status
- Modern UI: gradients, animations, mobile-friendly
## Structure
```
frontend/
├── src/
│   ├── components/   # Reusable UI components
│   ├── pages/        # Main pages (Home, History, Classes, Health)
│   ├── services/     # API client
│   ├── utils/        # Helper functions
│   ├── App.jsx       # Main app component
│   ├── main.jsx      # Entry point
│   └── index.css     # Tailwind & custom styles
├── public/           # Static assets
├── package.json      # NPM dependencies
├── tailwind.config.js
├── vite.config.js
└── README.md        # This file
```
## Quick Start

1. Install dependencies:
	```bash
	npm install
	```
2. Start development server:
	```bash
	npm run dev
	```
3. Access at [http://localhost:3000](http://localhost:3000)

## API Integration
- Backend must be running at `http://localhost:8000`
- API endpoints are proxied via Vite config

## Customization
- Update primary color in `tailwind.config.js`
- Add new pages/components in `src/pages` or `src/components`

## Environment Variables
- See `.env.example` for configuration

---
For backend setup, see `../backend/README.md`.
