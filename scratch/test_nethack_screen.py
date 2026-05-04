import pexpect
import pyte
import sys

def main():
    # 80x24 is standard nethack size
    screen = pyte.Screen(80, 24)
    stream = pyte.Stream(screen)
    
    # Start nethack
    child = pexpect.spawn("/opt/homebrew/bin/nethack", env={"TERM": "vt100"}, encoding='utf-8')
    child.setwinsize(24, 80)
    
    try:
        # Give it a second to start
        child.expect(pexpect.TIMEOUT, timeout=1)
    except:
        pass
        
    stream.feed(child.before if child.before else "")
    
    print("--- SCREEN START ---")
    for row in screen.display:
        print(row)
    print("--- SCREEN END ---")
    
    child.terminate()

if __name__ == "__main__":
    main()
