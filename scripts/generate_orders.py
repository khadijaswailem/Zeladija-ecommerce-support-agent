"""
scripts/generate_orders.py
Generates 200 synthetic orders for 40 customers using Faker.
Run: python scripts/generate_orders.py
Output: data/raw/orders.json
"""

from faker import Faker
import json
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path

fake = Faker()
random.seed(42)

STATUSES = [
    ("delivered", 0.40),
    ("in_transit", 0.25),
    ("delayed", 0.10),
    ("out_for_delivery", 0.10),
    ("returned", 0.08),
    ("cancelled", 0.07),
]

CARRIERS = ["UPS", "USPS", "FedEx", "DHL"]

# Zeladija Electronics product pool (aligned with product_catalog.json)
PRODUCTS = [
    {"sku": "ZLK-1001", "name": "Zeladija UltraView 4K Monitor 27\"", "price": 329.99},
    {"sku": "ZLK-1002", "name": "Zeladija SlimBook Pro 15", "price": 1099.99},
    {"sku": "ZLK-1003", "name": "Zeladija BudsAir Pro Wireless Earbuds", "price": 89.99},
    {"sku": "ZLK-1004", "name": "Zeladija MechKey RGB Mechanical Keyboard", "price": 129.99},
    {"sku": "ZLK-1005", "name": "Zeladija PrecisionPro Wireless Mouse", "price": 59.99},
    {"sku": "ZLK-1006", "name": "Zeladija ChargePad Max 15W Wireless Charger", "price": 34.99},
    {"sku": "ZLK-1007", "name": "Zeladija ShieldCam Pro Security Camera", "price": 79.99},
    {"sku": "ZLK-1008", "name": "Zeladija PowerBank Ultra 26800mAh", "price": 69.99},
    {"sku": "ZLK-1009", "name": "Zeladija SmartHub 7-in-1 USB-C Dock", "price": 49.99},
    {"sku": "ZLK-1010", "name": "Zeladija GamePad Elite Controller", "price": 74.99},
    {"sku": "ZLK-1011", "name": "Zeladija WebCam HD 1080p", "price": 44.99},
    {"sku": "ZLK-1012", "name": "Zeladija SSD Portable 1TB", "price": 89.99},
    {"sku": "ZLK-1017", "name": "Zeladija ProMic USB Condenser Microphone", "price": 79.99},
    {"sku": "ZLK-1020", "name": "Zeladija SlimBook Air 13", "price": 699.99},
]


def pick_status() -> str:
    """Randomly select an order status using weighted probabilities."""
    r = random.random()
    cumulative = 0.0
    for status, prob in STATUSES:
        cumulative += prob
        if r < cumulative:
            return status
    return "delivered"


def generate_order(customer_id: str) -> dict:
    """Generate a single realistic synthetic order for a given customer."""
    status = pick_status()
    order_date = fake.date_time_between(start_date="-90d", end_date="-1d")
    est_delivery = order_date + timedelta(days=random.randint(3, 10))

    num_items = random.randint(1, 3)
    chosen_products = random.sample(PRODUCTS, min(num_items, len(PRODUCTS)))
    items = [
        {
            "sku": p["sku"],
            "name": p["name"],
            "qty": random.randint(1, 2),
            "price": p["price"],
        }
        for p in chosen_products
    ]
    total = round(sum(i["price"] * i["qty"] for i in items), 2)

    return {
        "order_id": f"ORD-{uuid.uuid4().hex[:8].upper()}",
        "customer_id": customer_id,
        "status": status,
        "order_date": order_date.isoformat(),
        "estimated_delivery": est_delivery.isoformat(),
        "carrier": random.choice(CARRIERS),
        "tracking_url": f"https://track.Zeladija.com/{uuid.uuid4().hex[:12]}",
        "items": items,
        "total": total,
        "shipping_address": fake.address().replace("\n", ", "),
    }


def main():
    customers = [f"CUST-{i:04d}" for i in range(1, 41)]
    orders = []
    for cid in customers:
        orders.extend([generate_order(cid) for _ in range(5)])

    output_path = Path("data/raw/orders.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(orders, f, indent=2)

    print(f"Generated {len(orders)} orders for {len(customers)} customers")
    print(f"Saved to {output_path}")

    # Status distribution summary
    status_counts = {}
    for o in orders:
        status_counts[o["status"]] = status_counts.get(o["status"], 0) + 1
    print("\nStatus distribution:")
    for s, count in sorted(status_counts.items(), key=lambda x: -x[1]):
        print(f"  {s:20s}: {count}")


if __name__ == "__main__":
    main()