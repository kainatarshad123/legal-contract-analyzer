function Sidebar() {
  return (
    <aside className="w-64 min-h-screen bg-white border-r border-slate-200 px-5 py-6">
      <div className="mb-10">
        <h1 className="text-xl font-semibold text-slate-950">
          Legal Contract AI
        </h1>
        <p className="text-sm text-slate-500 mt-1">
          Contract review workspace
        </p>
      </div>

      <nav className="space-y-1">
        <button className="w-full text-left px-3 py-2.5 rounded-lg bg-slate-100 text-slate-950 font-medium">
          Contracts
        </button>

        <button className="w-full text-left px-3 py-2.5 rounded-lg text-slate-600 hover:bg-slate-100 hover:text-slate-950">
          Upload Contract
        </button>

        <button className="w-full text-left px-3 py-2.5 rounded-lg text-slate-600 hover:bg-slate-100 hover:text-slate-950">
          History
        </button>

        <button className="w-full text-left px-3 py-2.5 rounded-lg text-slate-600 hover:bg-slate-100 hover:text-slate-950">
          Settings
        </button>
      </nav>
    </aside>
  );
}

export default Sidebar;