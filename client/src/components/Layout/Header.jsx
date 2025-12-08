import './Header.css';

function Header() {
  return (
    <header className="header">
      <div className="header-left">
        <h1 className="header-title">TikTok Studio Scraper</h1>
      </div>
      <div className="header-right">
        <span className="header-version">v1.0.0</span>
      </div>
    </header>
  );
}

export default Header;
