function Header() {
  return (
    <header className="bg-[#f7f5f0] border-b border-slate-200">
      <div className="max-w-6xl mx-auto h-20 px-6 flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold tracking-tight">
            Legal Contract ML Agent
          </h2>
          <p className="text-sm text-slate-500">
            Contract review, clause detection, and risk signals
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button className="text-sm text-slate-600 hover:text-slate-950">
            Dashboard
          </button>

          <button className="bg-slate-950 text-white px-4 py-2 rounded-full text-sm font-medium hover:bg-slate-800">
            Upload
          </button>
        </div>
      </div>
    </header>
  );
}

export default Header;