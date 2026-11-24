"""Notices expire while paused and are shown once per application session."""
from pathlib import Path
from playwright.sync_api import sync_playwright


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.add_init_script("window.pywebview={api:{start:async()=>({ok:true}),poll:async()=>({events:[]})}}")
        page.goto((Path(__file__).resolve().parent / 'quantization_visualizer/web/index.html').as_uri())
        page.wait_for_function('!!window.lab')
        notice = "lab.receive({type:'status',status:'stopped',message:'Training stopped. Start begins a fresh run.'})"
        opacity = "getComputedStyle(document.getElementById('toast')).opacity"
        page.evaluate('lab.setPaused(true)')
        page.evaluate(notice)
        assert page.evaluate(opacity) == '1'
        # Repetition must not restart the five-second lifetime.
        page.evaluate(f'setTimeout(()=>{{{notice}}},1000)')
        page.wait_for_timeout(5100)
        assert page.evaluate(opacity) == '0', 'Paused notice did not expire'
        page.evaluate(notice)
        assert page.evaluate(opacity) == '0', 'Same-session notice reappeared'
        page.evaluate('lab.start()')
        page.evaluate(notice)
        assert page.evaluate(opacity) == '0', 'Starting another run erased notice history'
        page.evaluate("lab.receive({type:'status',status:'stopped',message:'A different notice'})")
        assert page.evaluate(opacity) == '1'
        page.reload()
        page.wait_for_function('!!window.lab')
        page.evaluate(notice)
        assert page.evaluate(opacity) == '1', 'Fresh app session suppressed its first notice'
        browser.close()
    print('PASS: five-second expiry during pause, duplicate suppression, run-reset persistence, fresh-session reset.')


if __name__ == '__main__':
    main()
