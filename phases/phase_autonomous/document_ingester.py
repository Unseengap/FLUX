"""
DocumentIngester — Process and store documents in FLUX memory.

Supports multiple formats:
- Text files (.txt, .md)
- PDF files
- JSON/CSV data
- Code files (.py, .js, etc.)
- HTML

Usage:
    ingester = DocumentIngester(flx_state, wave_dim=432)
    result = ingester.ingest(content, filename="doc.pdf")
    
    # List ingested documents
    docs = ingester.list_documents()
    
    # Query by filename
    chunks = ingester.get_chunks("doc.pdf")
"""

import json
import time
import hashlib
import base64
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import torch
from torch import Tensor


@dataclass
class DocumentChunk:
    """A chunk of document content."""
    chunk_id: str
    content: str
    source_file: str
    chunk_index: int
    total_chunks: int
    wave_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'chunk_id': self.chunk_id,
            'content': self.content[:200] + '...' if len(self.content) > 200 else self.content,
            'source_file': self.source_file,
            'chunk_index': self.chunk_index,
            'total_chunks': self.total_chunks,
            'wave_id': self.wave_id,
            'metadata': self.metadata,
        }


@dataclass
class IngestResult:
    """Result of document ingestion."""
    success: bool
    filename: str
    total_chunks: int
    wave_ids: List[str]
    content_hash: str
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'filename': self.filename,
            'total_chunks': self.total_chunks,
            'wave_ids': self.wave_ids,
            'content_hash': self.content_hash,
            'error': self.error,
            'metadata': self.metadata,
        }


class SemanticChunker:
    """Split text into semantic chunks."""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 100):
        """
        Initialize chunker.
        
        Args:
            chunk_size: Target chunk size in characters
            overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, text: str) -> List[str]:
        """
        Split text into chunks.
        
        Attempts to split at sentence boundaries when possible.
        """
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # Try to find sentence boundary
            best_break = end
            for sep in ['. ', '.\n', '! ', '? ', '\n\n', '\n']:
                idx = text.rfind(sep, start + self.chunk_size // 2, end)
                if idx > start:
                    best_break = idx + len(sep)
                    break
            
            chunks.append(text[start:best_break])
            start = best_break - self.overlap
        
        return chunks


class DocumentIngester:
    """
    Ingest documents into FLUX memory.
    
    Processes various document formats, chunks them semantically,
    encodes to waves, and stores in episodic memory.
    """
    
    # Supported file extensions
    TEXT_EXTENSIONS = {'.txt', '.md', '.rst', '.org'}
    CODE_EXTENSIONS = {'.py', '.js', '.ts', '.java', '.c', '.cpp', '.h', '.go', '.rs', '.rb'}
    DATA_EXTENSIONS = {'.json', '.csv', '.yaml', '.yml', '.toml'}
    MARKUP_EXTENSIONS = {'.html', '.htm', '.xml'}
    BINARY_EXTENSIONS = {'.pdf', '.docx', '.doc', '.pptx', '.xlsx'}
    
    def __init__(
        self,
        flx_state: Dict[str, Any],
        wave_dim: int = 432,
        chunk_size: int = 1000,
    ):
        """
        Initialize ingester.
        
        Args:
            flx_state: Loaded .flx state dict
            wave_dim: FLUX wave dimension
            chunk_size: Target chunk size
        """
        self.flx = flx_state
        self.wave_dim = wave_dim
        self.chunker = SemanticChunker(chunk_size)
        
        # Document storage
        self.documents: Dict[str, Dict[str, Any]] = flx_state.get('documents', {})
        
        # Wave cache
        self.wave_cache: Dict[str, Tensor] = {}
    
    def ingest(
        self,
        content: str,
        filename: str,
        chunk_size: Optional[int] = None,
    ) -> IngestResult:
        """
        Ingest a document.
        
        Args:
            content: Document content (text or base64 for binary)
            filename: Original filename
            chunk_size: Override default chunk size
            
        Returns:
            IngestResult with status and wave IDs
        """
        try:
            # Get file extension
            ext = Path(filename).suffix.lower()
            
            # Extract text based on format
            text = self._extract_text(content, ext)
            
            # Hash for deduplication
            content_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
            
            # Check if already ingested
            if content_hash in [d.get('content_hash') for d in self.documents.values()]:
                return IngestResult(
                    success=True,
                    filename=filename,
                    total_chunks=0,
                    wave_ids=[],
                    content_hash=content_hash,
                    metadata={'status': 'already_ingested'}
                )
            
            # Chunk
            if chunk_size:
                self.chunker.chunk_size = chunk_size
            chunks = self.chunker.chunk(text)
            
            # Encode and store each chunk
            wave_ids = []
            stored_chunks = []
            
            for i, chunk_text in enumerate(chunks):
                # Encode to wave
                wave = self._encode_chunk(chunk_text)
                wave_id = f"doc_{content_hash}_{i}"
                self.wave_cache[wave_id] = wave
                wave_ids.append(wave_id)
                
                # Create chunk object
                chunk = DocumentChunk(
                    chunk_id=f"{content_hash}_{i}",
                    content=chunk_text,
                    source_file=filename,
                    chunk_index=i,
                    total_chunks=len(chunks),
                    wave_id=wave_id,
                    metadata={
                        'char_count': len(chunk_text),
                        'ingested_at': datetime.now().isoformat(),
                    }
                )
                stored_chunks.append(chunk.to_dict())
                
                # Store in episodic memory
                self._store_in_memory(chunk)
            
            # Save document record
            self.documents[filename] = {
                'filename': filename,
                'content_hash': content_hash,
                'total_chunks': len(chunks),
                'wave_ids': wave_ids,
                'ingested_at': datetime.now().isoformat(),
                'file_type': ext,
                'total_chars': len(text),
                'chunks': stored_chunks,
            }
            
            # Persist to flx
            self.flx['documents'] = self.documents
            
            return IngestResult(
                success=True,
                filename=filename,
                total_chunks=len(chunks),
                wave_ids=wave_ids,
                content_hash=content_hash,
                metadata={
                    'file_type': ext,
                    'total_chars': len(text),
                }
            )
            
        except Exception as e:
            return IngestResult(
                success=False,
                filename=filename,
                total_chunks=0,
                wave_ids=[],
                content_hash='',
                error=str(e)
            )
    
    def _extract_text(self, content: str, ext: str) -> str:
        """Extract text from content based on file type."""
        if ext in self.TEXT_EXTENSIONS or ext in self.CODE_EXTENSIONS:
            return content
        
        if ext in self.DATA_EXTENSIONS:
            return self._extract_from_data(content, ext)
        
        if ext in self.MARKUP_EXTENSIONS:
            return self._extract_from_markup(content)
        
        if ext in self.BINARY_EXTENSIONS:
            return self._extract_from_binary(content, ext)
        
        # Default: treat as text
        return content
    
    def _extract_from_data(self, content: str, ext: str) -> str:
        """Extract text from data files."""
        if ext == '.json':
            try:
                data = json.loads(content)
                return json.dumps(data, indent=2)
            except:
                return content
        
        if ext == '.csv':
            lines = content.split('\n')
            # Convert CSV to readable format
            result = []
            for i, line in enumerate(lines[:100]):  # Limit rows
                result.append(f"Row {i}: {line}")
            return '\n'.join(result)
        
        return content
    
    def _extract_from_markup(self, content: str) -> str:
        """Extract text from HTML/XML."""
        import re
        # Simple HTML tag stripping
        text = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _extract_from_binary(self, content: str, ext: str) -> str:
        """Extract text from binary files (PDF, DOCX)."""
        # Try base64 decode
        try:
            binary = base64.b64decode(content)
        except:
            # Not base64, might be already decoded or text
            return content
        
        if ext == '.pdf':
            return self._extract_pdf(binary)
        
        if ext in {'.docx', '.doc'}:
            return self._extract_docx(binary)
        
        return f"[Binary file: {ext}, {len(binary)} bytes]"
    
    def _extract_pdf(self, binary: bytes) -> str:
        """Extract text from PDF bytes."""
        try:
            # Try PyMuPDF
            import fitz
            doc = fitz.open(stream=binary, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            return text
        except ImportError:
            pass
        
        try:
            # Try pdfplumber
            import pdfplumber
            import io
            with pdfplumber.open(io.BytesIO(binary)) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
            return text
        except ImportError:
            pass
        
        return "[PDF extraction not available - install pymupdf or pdfplumber]"
    
    def _extract_docx(self, binary: bytes) -> str:
        """Extract text from DOCX bytes."""
        try:
            import docx
            import io
            doc = docx.Document(io.BytesIO(binary))
            return '\n'.join(p.text for p in doc.paragraphs)
        except ImportError:
            return "[DOCX extraction not available - install python-docx]"
    
    def _encode_chunk(self, text: str) -> Tensor:
        """Encode chunk to wave via CSE."""
        cse_state = self.flx.get('cse', {})
        
        # Simplified encoding - full implementation uses CSE weights
        byte_seq = list(text.encode('utf-8'))
        wave = torch.randn(self.wave_dim)
        
        return wave
    
    def _store_in_memory(self, chunk: DocumentChunk):
        """Store chunk in episodic memory."""
        memory = self.flx.setdefault('memory', {})
        state_dict = memory.setdefault('state_dict', {})
        episodic = state_dict.setdefault('episodic', {})
        metadata = episodic.setdefault('metadata', [])
        
        metadata.append({
            'id': len(metadata),
            'content': chunk.content[:500],  # Truncate for storage
            'importance': 0.7,
            'source': chunk.source_file,
            'chunk_id': chunk.chunk_id,
            'wave_id': chunk.wave_id,
            'timestamp': time.time(),
        })
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all ingested documents."""
        return [
            {
                'filename': doc['filename'],
                'total_chunks': doc['total_chunks'],
                'ingested_at': doc['ingested_at'],
                'total_chars': doc.get('total_chars', 0),
            }
            for doc in self.documents.values()
        ]
    
    def get_chunks(self, filename: str) -> List[Dict[str, Any]]:
        """Get chunks for a document."""
        doc = self.documents.get(filename)
        if doc is None:
            return []
        return doc.get('chunks', [])
    
    def get_wave(self, wave_id: str) -> Optional[Tensor]:
        """Get a document wave."""
        return self.wave_cache.get(wave_id)
    
    def delete_document(self, filename: str) -> bool:
        """Delete a document and its chunks."""
        if filename not in self.documents:
            return False
        
        doc = self.documents[filename]
        
        # Remove waves from cache
        for wave_id in doc.get('wave_ids', []):
            self.wave_cache.pop(wave_id, None)
        
        # Remove from documents
        del self.documents[filename]
        self.flx['documents'] = self.documents
        
        return True
    
    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search across all ingested documents."""
        query_lower = query.lower()
        results = []
        
        for doc in self.documents.values():
            for chunk in doc.get('chunks', []):
                content = chunk.get('content', '')
                if query_lower in content.lower():
                    results.append({
                        'filename': doc['filename'],
                        'chunk_id': chunk['chunk_id'],
                        'content': content[:300],
                        'relevance': content.lower().count(query_lower),
                    })
        
        # Sort by relevance
        results.sort(key=lambda x: x['relevance'], reverse=True)
        return results[:limit]
