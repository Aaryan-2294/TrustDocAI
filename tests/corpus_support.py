"""Small multi-file Python/Markdown corpus for retrieval tests (no heavy imports)."""

from __future__ import annotations


def corpus_files(extra_modules: int = 8) -> dict[str, bytes]:
    files = {
        "billing.py": b'''"""Billing helpers."""


def calculate_invoice_total(items, tax_rate=0.2):
    """Sum item prices and apply tax_rate."""
    return sum(items) * (1 + tax_rate)


class InvoiceRenderer:
    """Turns invoices into printable output."""

    def render_pdf(self, invoice):
        """Render an invoice as a PDF document."""
        return b"%PDF"
''',
        "users.py": b'''def create_user(name, email):
    """Create a user account."""
    return {"name": name, "email": email}


def delete_user(user_id):
    """Remove a user account permanently."""
    return True
''',
        "cache.py": b'''class LruCache:
    """Least recently used cache with a fixed capacity."""

    def get(self, key):
        """Return the cached value for key or None."""
        return None

    def put(self, key, value):
        """Store a value, evicting the oldest entry when full."""
        return None
''',
        "mailer.py": b'''def send_email(recipient, subject, body, retries=3):
    """Send an email, retrying on transient SMTP failures."""
    return True
''',
        "geometry.py": b'''import math


def circle_area(radius):
    """Area of a circle with the given radius."""
    return math.pi * radius ** 2
''',
        "README.md": b"""# Shop

A tiny shop library.

## Installation

Install with pip and import the modules you need.

## Usage

Create users, calculate totals and send receipts by email.
""",
    }
    for i in range(extra_modules):
        files[f"filler_{i}.py"] = (
            f'def filler_task_{i}(value_{i}):\n    """Unrelated utility number {i}."""\n    return value_{i}\n'
        ).encode()
    return files
