#!/usr/bin/env python3
"""Static file server + POST collector for the webml trace experiment.
Serves the current directory; POST /log appends to trace.log, POST /trace
writes trace.json. Port 8917."""
import http.server
import os
import sys

DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(DIR)

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(n)
        if self.path == '/trace':
            with open('trace.json', 'wb') as f:
                f.write(body)
        else:
            with open('trace.log', 'ab') as f:
                f.write(body + b'\n')
        self.send_response(200)
        self.send_header('Content-Length', '0')
        self.end_headers()

    def log_message(self, fmt, *args):
        pass  # quiet

print('collector on :8917', flush=True)
http.server.ThreadingHTTPServer(('127.0.0.1', 8917), Handler).serve_forever()
