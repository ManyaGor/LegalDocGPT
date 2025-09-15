# LegalDocGPT UI

A modern, responsive web interface for LegalDocGPT - an AI-powered legal document analysis tool.

## Features

- 🚀 **Modern Design**: Clean, professional interface with glass morphism effects
- 📱 **Responsive**: Optimized for desktop, tablet, and mobile devices
- 🎨 **Beautiful UI**: Gradient backgrounds, smooth animations, and modern components
- 📁 **Drag & Drop**: Easy file upload with drag and drop support
- ⚡ **Real-time Progress**: Progress bars and loading states for better UX
- 🔒 **Secure**: Client-side processing with secure file handling
- 🎯 **Accessible**: WCAG compliant with proper focus management

## Tech Stack

- **Framework**: Next.js 15 with App Router
- **Styling**: Tailwind CSS 4
- **Language**: TypeScript
- **Components**: Custom reusable UI components
- **Icons**: Heroicons (SVG)
- **Fonts**: Geist Sans & Geist Mono

## Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn
- LegalDocGPT backend running on port 8000

### Installation

1. **Navigate to the UI directory:**
   ```bash
   cd legaldoc-ui
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Start the development server:**
   ```bash
   npm run dev
   ```

4. **Open your browser:**
   Navigate to [http://localhost:3000](http://localhost:3000)

### Available Scripts

- `npm run dev` - Start development server with Turbopack
- `npm run build` - Build for production
- `npm run start` - Start production server

## UI Components

The UI includes several reusable components:

- **Button**: Primary, secondary, ghost, and destructive variants
- **Card**: Flexible card component with header, content, and description
- **ProgressBar**: Animated progress indicators
- **Alert**: Success, error, warning, and info alerts

## Design System

### Colors
- **Primary**: Blue gradient (#3B82F6 to #6366F1)
- **Secondary**: Indigo (#6366F1)
- **Success**: Green (#10B981)
- **Error**: Red (#EF4444)
- **Warning**: Yellow (#F59E0B)
- **Info**: Blue (#3B82F6)

### Typography
- **Headings**: Geist Sans (bold weights)
- **Body**: Geist Sans (regular)
- **Code**: Geist Mono

### Spacing
- Consistent 4px grid system
- Responsive padding and margins
- Proper component spacing

## Features in Detail

### File Upload
- Drag and drop support
- File type validation (PDF, DOCX)
- File size display
- Visual feedback for drag states

### Document Processing
- Real-time progress tracking
- Loading states with spinners
- Error handling with user-friendly messages
- Success indicators

### Results Display
- Numbered point system
- Clean typography
- Visual hierarchy
- Download functionality

## Responsive Design

The UI is fully responsive with breakpoints:
- **Mobile**: < 640px
- **Tablet**: 640px - 1024px  
- **Desktop**: > 1024px

## Accessibility

- WCAG 2.1 AA compliant
- Keyboard navigation support
- Screen reader friendly
- High contrast mode support
- Reduced motion support

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Development

### Project Structure
```
legaldoc-ui/
├── app/                 # Next.js app directory
│   ├── globals.css     # Global styles
│   ├── layout.tsx      # Root layout
│   └── page.tsx        # Home page
├── components/         # Reusable components
│   └── ui/             # UI component library
├── public/             # Static assets
└── package.json        # Dependencies
```

### Adding New Components

1. Create component in `components/ui/`
2. Export from `components/ui/index.ts`
3. Import and use in pages

### Styling Guidelines

- Use Tailwind CSS classes
- Follow the design system
- Use semantic color names
- Maintain consistent spacing
- Test on multiple screen sizes

## Deployment

### Production Build
```bash
npm run build
npm run start
```

### Environment Variables
No environment variables required for basic functionality.

## Troubleshooting

### Common Issues

1. **Backend Connection Error**
   - Ensure LegalDocGPT backend is running on port 8000
   - Check CORS settings if needed

2. **File Upload Issues**
   - Verify file type (PDF or DOCX only)
   - Check file size (max 50MB)
   - Ensure stable internet connection

3. **Build Errors**
   - Clear node_modules and reinstall
   - Check Node.js version compatibility
   - Verify TypeScript configuration

## Contributing

1. Follow the existing code style
2. Use TypeScript for type safety
3. Test on multiple devices
4. Ensure accessibility compliance
5. Update documentation as needed

## License

This project is part of LegalDocGPT and follows the same licensing terms.