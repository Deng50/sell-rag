from __future__ import annotations

from pathlib import Path

from sell_rag.domain import Role
from sell_rag.ingestion.pipeline import ParentChildChunker, ParsedElement, parse_header_footer_candidates


def test_repeated_margin_text_is_removed() -> None:
    elements = []
    for page in range(1, 6):
        elements.extend([
            ParsedElement("text", text=f"公司资料 {page}", page_no=page, bbox=[0, 0.01, 1, 0.04]),
            ParsedElement("text", text=f"第{page}页正文", page_no=page, bbox=[0, 0.2, 1, 0.7]),
        ])
    result = parse_header_footer_candidates(elements)
    assert len(result) == 5
    assert all("正文" in item.text for item in result)


def test_xlsx_products_are_structured(services, tmp_path: Path) -> None:
    from openpyxl import Workbook

    path = tmp_path / "products.xlsx"
    book = Workbook()
    sheet = book.active
    sheet.title = "商品"
    sheet.append(["商品编号", "商品名称", "类别", "价格", "库存", "特点"])
    sheet.append(["W-1", "测试水", "饮品", 2.5, 3, "清爽"])
    book.save(path)
    result = services.ingestion.ingest(path)
    products = services.database.query_products(text="测试水")
    assert result["products"] == 1
    assert products[0].price.as_tuple().exponent == -2
    assert products[0].stock == 3


def test_xlsx_multirow_headers(services, tmp_path: Path) -> None:
    from openpyxl import Workbook

    path = tmp_path / "multi.xlsx"
    book = Workbook()
    sheet = book.active
    sheet.append(["基础信息", None, "销售信息", None])
    sheet.append(["商品编号", "商品名称", "价格", "库存"])
    sheet.append(["M-1", "多行表头商品", 9.9, 4])
    book.save(path)
    services.ingestion.ingest(path)
    result = services.database.query_products(text="多行表头商品")
    assert result[0].sku == "M-1"
    assert result[0].stock == 4


def test_chunker_does_not_cross_element_boundaries() -> None:
    from sell_rag.domain import DocumentElement

    elements = [
        DocumentElement(element_id="a", document_id="d", document_version=1, order=0,
                        content_type="text", text="甲" * 40, title="A"),
        DocumentElement(element_id="b", document_id="d", document_version=1, order=1,
                        content_type="text", text="乙" * 40, title="B"),
    ]
    chunks = ParentChildChunker(child_tokens=10, parent_tokens=20).chunk("d", 1, [Role.GUEST], elements)
    assert all(not ("甲" in chunk.text and "乙" in chunk.text) for chunk in chunks)
