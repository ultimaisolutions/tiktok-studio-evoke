import { useState } from 'react';
import './App.css';
import Header from './components/Layout/Header';
import Sidebar from './components/Layout/Sidebar';
import DownloadView from './components/Views/DownloadView';
import StudioView from './components/Views/StudioView';
import AnalysisView from './components/Views/AnalysisView';
import VideosView from './components/Views/VideosView';

function App() {
  const [activeView, setActiveView] = useState('download');

  const renderView = () => {
    switch (activeView) {
      case 'download':
        return <DownloadView />;
      case 'studio':
        return <StudioView />;
      case 'analysis':
        return <AnalysisView />;
      case 'videos':
        return <VideosView />;
      default:
        return <DownloadView />;
    }
  };

  return (
    <div className="app">
      <Header />
      <div className="app-body">
        <Sidebar activeView={activeView} onViewChange={setActiveView} />
        <main className="main-content">
          {renderView()}
        </main>
      </div>
    </div>
  );
}

export default App;
