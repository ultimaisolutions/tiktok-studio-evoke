import './Sidebar.css';

const menuItems = [
  { id: 'download', label: 'Download', icon: '↓' },
  { id: 'studio', label: 'Studio', icon: '◎' },
  { id: 'analysis', label: 'Analysis', icon: '◈' },
  { id: 'videos', label: 'Videos', icon: '▶' },
  { id: 'extractor', label: 'API Extract', icon: '⚡' },
];

function Sidebar({ activeView, onViewChange }) {
  return (
    <aside className="sidebar">
      <nav className="sidebar-nav">
        {menuItems.map((item) => (
          <button
            key={item.id}
            className={`sidebar-item ${activeView === item.id ? 'active' : ''}`}
            onClick={() => onViewChange(item.id)}
          >
            <span className="sidebar-icon">{item.icon}</span>
            <span className="sidebar-label">{item.label}</span>
          </button>
        ))}
      </nav>
    </aside>
  );
}

export default Sidebar;
