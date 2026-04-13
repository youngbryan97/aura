import sys
import json
import traceback
import io
import contextlib
import os

def main():
    namespace = {}
    sys.stdout.reconfigure(line_buffering=True)
    
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            
            line = line.strip()
            if not line:
                continue
            
            try:
                size = int(line)
            except ValueError:
                continue
                
            code = sys.stdin.read(size)
            
            out = io.StringIO()
            success = False
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                try:
                    # Execute the code in the shared namespace
                    exec(code, namespace)
                    success = True
                except BaseException as e:
                    traceback.print_exc(file=out)
            
            result_text = out.getvalue()
            # Send result back
            resp = json.dumps({"success": success, "output": result_text})
            sys.stdout.write(f"{len(resp)}\n{resp}\n")
            sys.stdout.flush()
            
        except Exception as _e:
            err = json.dumps({"success": False, "output": f"Daemon Error: {str(_e)}"})
            sys.stdout.write(f"{len(err)}\n{err}\n")
            sys.stdout.flush()

if __name__ == "__main__":
    main()
