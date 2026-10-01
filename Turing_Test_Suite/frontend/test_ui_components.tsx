import { render, screen, fireEvent, cleanup, waitFor } from '@testing-library/react';
import React from 'react';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SRC_ROOT = path.resolve(HERE, '../../src');

const modalSrc = fs.readFileSync(path.join(SRC_ROOT, 'components/ui/Modal.tsx'), 'utf8');
const buttonSrc = fs.readFileSync(path.join(SRC_ROOT, 'components/ui/Button.tsx'), 'utf8');
const cardSrc = fs.readFileSync(path.join(SRC_ROOT, 'components/ui/Card.tsx'), 'utf8');
const badgeSrc = fs.readFileSync(path.join(SRC_ROOT, 'components/ui/Badge.tsx'), 'utf8');
const inputSrc = fs.readFileSync(path.join(SRC_ROOT, 'components/ui/Input.tsx'), 'utf8');

async function importModal() {
  return (await import('@/components/ui/Modal')).Modal;
}
async function importButton() {
  return (await import('@/components/ui/Button')).Button;
}
async function importCard() {
  return (await import('@/components/ui/Card'));
}
async function importBadge() {
  return (await import('@/components/ui/Badge')).Badge;
}
async function importInput() {
  return (await import('@/components/ui/Input')).Input;
}

describe('Modal', () => {
  beforeEach(() => vi.resetModules());

  /**
   * SWARM-2 BUG #15 — when the modal closes, the cleanup restores
   * `document.body.style.overflow` to `'unset'` (which OVERWRITES the
   * CSS-defined value), not to `''` (which RESTORES the CSS-defined value).
   *
   * We open + close the modal and assert that body.style.overflow is `''`,
   * not `'unset'`.
   */
  test('test_cleanup_restores_css_overflow_not_unset', async () => {
    const Modal = await importModal();
    // Simulate the global CSS that the project relies on
    // (globals.css sets `body { overflow: hidden; }`).
    // The CSS isn't loaded in jsdom, but the assertion is on the inline style.
    const { rerender, unmount } = render(
      <Modal isOpen={true} onClose={() => {}} title="Test">
        Body
      </Modal>,
    );
    // While open, the inline `overflow: hidden` is set.
    expect(document.body.style.overflow).toBe('hidden');
    // Close by re-rendering with isOpen=false. The Modal returns null when
    // closed; the effect cleanup runs and sets overflow back.
    rerender(
      <Modal isOpen={false} onClose={() => {}} title="Test">
        Body
      </Modal>,
    );
    // Buggy code sets `'unset'`; fix sets `''`.
    expect(document.body.style.overflow).not.toBe('unset');
    expect(document.body.style.overflow).toBe('');
    unmount();
  });

  /**
   * SWARM-2 BUG #16 — modal overlay must have role="dialog".
   */
  test('test_has_role_dialog', async () => {
    const Modal = await importModal();
    render(
      <Modal isOpen={true} onClose={() => {}} title="Test">
        Body
      </Modal>,
    );
    // The overlay container is the element with role="dialog".
    expect(screen.getByRole('dialog')).toBeTruthy();
  });

  /**
   * SWARM-2 BUG #16 — aria-modal="true".
   */
  test('test_has_aria_modal_true', async () => {
    const Modal = await importModal();
    render(
      <Modal isOpen={true} onClose={() => {}} title="Test">
        Body
      </Modal>,
    );
    const dialog = screen.getByRole('dialog');
    expect(dialog.getAttribute('aria-modal')).toBe('true');
  });

  /**
   * SWARM-2 BUG #16 — aria-labelledby pointing to the title.
   */
  test('test_has_aria_labelledby', async () => {
    const Modal = await importModal();
    render(
      <Modal isOpen={true} onClose={() => {}} title="My Modal Title">
        Body
      </Modal>,
    );
    const dialog = screen.getByRole('dialog');
    const labelledby = dialog.getAttribute('aria-labelledby');
    expect(labelledby).toBeTruthy();
    // The element referenced by aria-labelledby should exist and contain
    // the title text.
    const label = document.getElementById(labelledby!);
    expect(label).toBeTruthy();
    expect((label!.textContent || '').includes('My Modal Title')).toBe(true);
  });

  /**
   * SWARM-2 BUG #16 — Tab key cycles focus within the modal, not to background.
   *
   * We render the modal with two focusable elements (the close button + a
   * link) plus some background buttons. We focus the first focusable in the
   * modal, then press Tab repeatedly, and assert that focus never leaves
   * the modal's elements.
   */
  test('test_focus_trapped_inside_modal', async () => {
    const Modal = await importModal();
    render(
      <div>
        <button>background button</button>
        <Modal isOpen={true} onClose={() => {}} title="Test">
          <button>inside button 1</button>
          <button>inside button 2</button>
        </Modal>
      </div>,
    );
    // Find the modal dialog.
    const dialog = screen.getByRole('dialog');
    // Tab from the first focusable element inside the dialog.
    const focusables = dialog.querySelectorAll('button, a, input, textarea, select, [tabindex]');
    expect(focusables.length).toBeGreaterThan(0);
    const first = focusables[0] as HTMLElement;
    first.focus();
    expect(document.activeElement).toBe(first);

    // Press Tab several times — focus must stay inside the dialog.
    for (let i = 0; i < focusables.length + 3; i++) {
      fireEvent.keyDown(dialog, { key: 'Tab', bubbles: true });
      // The active element should be inside the dialog.
      expect(dialog.contains(document.activeElement)).toBe(true);
    }
  });

  /**
   * SWARM-2 BUG #16 — pressing Escape calls onClose.
   */
  test('test_escape_key_closes_modal', async () => {
    const Modal = await importModal();
    const onClose = vi.fn();
    render(
      <Modal isOpen={true} onClose={onClose} title="Test">
        Body
      </Modal>,
    );
    fireEvent.keyDown(document, { key: 'Escape', bubbles: true });
    expect(onClose).toHaveBeenCalled();
  });
});

describe('Button', () => {
  beforeEach(() => vi.resetModules());

  /**
   * SWARM-2 BUG #18 — `<Button>` defaults to `type="submit"` (the HTML default),
   * so clicking it inside a `<form>` submits the form. The fix sets
   * `type="button"` as the default.
   *
   * We render a form containing a Button, click the button, and assert
   * the form was NOT submitted (no `onSubmit` callback fired).
   */
  test('test_default_type_is_button_not_submit', async () => {
    const Button = await importButton();
    const onSubmit = vi.fn();
    render(
      <form onSubmit={onSubmit}>
        <Button>Click</Button>
      </form>,
    );
    const btn = screen.getByText('Click');
    fireEvent.click(btn);
    // If type="submit" (the HTML default), the form's onSubmit fires.
    // If type="button" (the fix), the form does NOT submit.
    expect(onSubmit).not.toHaveBeenCalled();
  });
});

describe('Card', () => {
  beforeEach(() => vi.resetModules());

  /**
   * SWARM-2 BUG #17 — `CardTitle`'s `forwardRef` type is `HTMLParagraphElement`
   * but the rendered element is `<h3>` (HTMLHeadingElement). Consumers using
   * the ref would have a type mismatch.
   *
   * Static check: the `forwardRef` call must specify `HTMLHeadingElement`,
   * not `HTMLParagraphElement`.
   */
  test('test_cardtitle_ref_type_matches_dom_element', () => {
    // Look for the forwardRef call signature for CardTitle.
    // Buggy: `React.forwardRef<HTMLParagraphElement, ...>`
    // Fixed: `React.forwardRef<HTMLHeadingElement, ...>`
    const cardTitleIdx = cardSrc.indexOf('CardTitle = React.forwardRef');
    expect(cardTitleIdx).toBeGreaterThan(-1);
    // Read 150 chars after the forwardRef call.
    const block = cardSrc.slice(cardTitleIdx, cardTitleIdx + 150);
    expect(block).toContain('HTMLHeadingElement');
    expect(block).not.toContain('HTMLParagraphElement');
  });
});

describe('Badge', () => {
  beforeEach(() => vi.resetModules());

  /**
   * SWARM-2 BUG #19 — Badge is rendered as a `<div>`, which is invalid inside
   * inline contexts (e.g., a `<p>` cannot contain a `<div>` per HTML spec).
   * The fix renders it as a `<span>`.
   */
  test('test_renders_as_span_not_div', async () => {
    const Badge = await importBadge();
    const { container } = render(<Badge>Test</Badge>);
    expect(container.firstChild).not.toBeNull();
    expect(container.firstChild!.nodeName.toLowerCase()).toBe('span');
  });
});

describe('Input', () => {
  beforeEach(() => vi.resetModules());

  /**
   * Input must accept `aria-invalid` (passed through via `...props`).
   */
  test('test_supports_aria_invalid', async () => {
    const Input = await importInput();
    render(<Input aria-invalid="true" />);
    const input = document.querySelector('input');
    expect(input).toBeTruthy();
    expect(input!.getAttribute('aria-invalid')).toBe('true');
  });

  /**
   * Input must accept `aria-describedby` (passed through via `...props`).
   */
  test('test_supports_aria_describedby', async () => {
    const Input = await importInput();
    render(<Input aria-describedby="hint-id" />);
    const input = document.querySelector('input');
    expect(input).toBeTruthy();
    expect(input!.getAttribute('aria-describedby')).toBe('hint-id');
  });
});