"use client";

import React, { useState, useRef } from 'react';
import { Badge } from '../ui/Badge';

interface FileUploaderProps {
  /**
   * Called whenever the user picks a new file.
   * Pass `null` when the selection is cleared.
   * The parent owns the file state and decides when/how to upload it.
   */
  onFileSelect?: (file: File | null) => void;
  /** Controlled: highlight the uploader as "file selected" from outside. */
  selectedFileName?: string | null;
}

export const FileUploader: React.FC<FileUploaderProps> = ({
  onFileSelect,
  selectedFileName,
}) => {
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] ?? null;
    setError(null);

    if (!file) {
      onFileSelect?.(null);
      return;
    }

    const allowedExtensions = ['.csv', '.xlsx', '.xls', '.txt', '.json', '.pdf', '.md'];
    const ext = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!allowedExtensions.includes(ext)) {
      setError(`Unsupported file type: ${ext}`);
      onFileSelect?.(null);
      if (fileInputRef.current) fileInputRef.current.value = '';
      return;
    }

    onFileSelect?.(file);
  };

  const hasFile = Boolean(selectedFileName);

  return (
    <div className="w-full flex flex-col bg-white border-b-2 border-black p-6 relative">
      <div className="font-mono text-xs font-bold tracking-widest text-gray-500 mb-4 shrink-0">
        STEP 0: UPLOAD DATASET
      </div>

      <div className="relative group shrink-0">
        <input
          type="file"
          accept=".csv,.xlsx,.xls,.txt,.json,.pdf,.md"
          ref={fileInputRef}
          onChange={handleFileChange}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
        />
        <div
          className={`w-full flex flex-col items-center justify-center border-2 border-dashed p-8 transition-colors duration-200
            ${hasFile
              ? 'border-green-500 bg-green-50'
              : 'border-gray-300 group-hover:border-black bg-[#F9FAFB] group-hover:bg-gray-100'
            }`}
        >
          <span className="font-mono text-sm font-semibold text-black tracking-widest mb-2">
            {hasFile ? `✓ ${selectedFileName}` : 'DRAG & DROP OR CLICK TO BROWSE'}
          </span>
          <span className="font-mono text-[10px] text-gray-500">
            ACCEPTED FORMATS: CSV, EXCEL, TXT, JSON, PDF, MD
          </span>
        </div>
      </div>

      {error && (
        <div className="mt-4 font-mono text-[10px] text-red-500 font-bold bg-red-50 p-2 border border-red-200">
          SYSTEM ERROR: {error}
        </div>
      )}

      {hasFile && !error && (
        <div className="mt-4 font-mono text-[10px] text-green-700 font-bold bg-green-50 p-2 border border-green-200 flex items-center justify-between animate-fade-in">
          <span>DATASET SELECTED. READY FOR INGESTION.</span>
          <Badge variant="success">✅</Badge>
        </div>
      )}
    </div>
  );
};
