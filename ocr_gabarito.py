#!/usr/bin/env python3
"""
Leitor OCR de Gabarito - OBMEP Cartão-Resposta
Detecta círculos preenchidos em um cartão-resposta de 20 questões (A-E).
"""

import cv2
import numpy as np
import sys
import json
import argparse
from pathlib import Path


# ─── Configuração do grid ────────────────────────────────────────────────────
ALTERNATIVAS = ["A", "B", "C", "D", "E"]
NUM_QUESTOES = 20
# Fração mínima de pixels escuros para considerar um círculo marcado
FILL_THRESHOLD = 0.35


def order_points(pts):
    """Ordena 4 pontos: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    """Aplica transformação de perspectiva para retificar o cartão."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))


def detect_card_contour(gray):
    """Detecta o contorno retangular principal do cartão."""
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blurred, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edged = cv2.dilate(edged, kernel, iterations=2)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for c in contours[:5]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            img_area = gray.shape[0] * gray.shape[1]
            if area > img_area * 0.15:
                return approx.reshape(4, 2).astype("float32")
    return None


def find_answer_grid(warped_gray):
    """
    Localiza a região do grid de respostas (A-E × 1-20) dentro do cartão retificado.
    Retorna (x, y, w, h) da região do grid, já pulando a linha de cabeçalho com os
    números das questões (1-20).
    """
    h, w = warped_gray.shape
    # A área de respostas (incluindo linha de números) ocupa a região central-inferior
    # Layout OBMEP: linha de números + 5 linhas A-E = 6 linhas no total
    grid_y_start = int(h * 0.52)
    grid_y_end = int(h * 0.82)
    grid_x_start = int(w * 0.02)
    grid_x_end = int(w * 0.93)

    full_h = grid_y_end - grid_y_start
    # Cada linha (incluindo a de números) tem altura ≈ full_h/6
    header_row_h = int(full_h / 6)

    # Pula a linha de cabeçalho (números 1-20)
    answer_y_start = grid_y_start + header_row_h
    answer_y_end = grid_y_end

    return grid_x_start, answer_y_start, grid_x_end - grid_x_start, answer_y_end - answer_y_start


def analyze_circle(cell_gray):
    """
    Determina se um círculo está marcado (preenchido).
    Retorna True se marcado, False se vazio.
    """
    h, w = cell_gray.shape
    # Usa apenas a região central para evitar bordas do grid
    margin_x = int(w * 0.15)
    margin_y = int(h * 0.15)
    region = cell_gray[margin_y:h - margin_y, margin_x:w - margin_x]
    if region.size == 0:
        return False, 0.0
    _, binary = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    fill_ratio = np.count_nonzero(binary) / binary.size
    return fill_ratio >= FILL_THRESHOLD, fill_ratio


def read_answers(image_path: str, gabarito: list = None, debug: bool = False) -> dict:
    """
    Lê o gabarito de uma imagem de cartão-resposta.

    Args:
        image_path: Caminho para a imagem do cartão.
        gabarito: Lista com as respostas corretas (ex: ["A","B","C",...]).
        debug: Salva imagem de depuração com anotações.

    Returns:
        dict com 'respostas', 'score' (se gabarito fornecido), e 'debug_image' path.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Não foi possível carregar a imagem: {image_path}")

    orig = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Detectar e retificar o cartão
    card_pts = detect_card_contour(gray)
    if card_pts is not None:
        warped = four_point_transform(img, card_pts)
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    else:
        # Sem detecção de bordas, usa a imagem inteira
        print("[AVISO] Borda do cartão não detectada. Usando imagem completa.")
        warped = img.copy()
        warped_gray = gray.copy()

    wh, ww = warped_gray.shape

    # 2. Localizar o grid de respostas
    gx, gy, gw, gh = find_answer_grid(warped_gray)
    grid_gray = warped_gray[gy:gy + gh, gx:gx + gw]
    grid_color = warped[gy:gy + gh, gx:gx + gw]

    if debug:
        dbg = warped.copy()
        cv2.rectangle(dbg, (gx, gy), (gx + gw, gy + gh), (0, 255, 0), 2)

    # 3. Dividir o grid em células (5 linhas × 20 colunas)
    cell_h = gh // len(ALTERNATIVAS)
    cell_w = gw // NUM_QUESTOES

    respostas = {}
    fill_matrix = np.zeros((len(ALTERNATIVAS), NUM_QUESTOES))

    for row_idx, alt in enumerate(ALTERNATIVAS):
        for col_idx in range(NUM_QUESTOES):
            y1 = row_idx * cell_h
            y2 = y1 + cell_h
            x1 = col_idx * cell_w
            x2 = x1 + cell_w
            cell = grid_gray[y1:y2, x1:x2]
            marked, ratio = analyze_circle(cell)
            fill_matrix[row_idx, col_idx] = ratio

            if debug:
                cx = gx + x1 + cell_w // 2
                cy = gy + y1 + cell_h // 2
                color = (0, 0, 255) if marked else (255, 0, 0)
                cv2.circle(dbg, (cx, cy), min(cell_w, cell_h) // 3, color, 2)

    # 4. Para cada questão, escolhe a alternativa com maior preenchimento
    for col_idx in range(NUM_QUESTOES):
        col_fills = fill_matrix[:, col_idx]
        best_row = int(np.argmax(col_fills))
        best_fill = col_fills[best_row]
        questao_num = col_idx + 1
        if best_fill >= FILL_THRESHOLD:
            respostas[questao_num] = ALTERNATIVAS[best_row]
        else:
            respostas[questao_num] = "?"  # não marcado

    # 5. Anotar debug
    debug_path = None
    if debug:
        for col_idx in range(NUM_QUESTOES):
            resp = respostas.get(col_idx + 1, "?")
            if resp != "?":
                row_idx = ALTERNATIVAS.index(resp)
                cx = gx + col_idx * cell_w + cell_w // 2
                cy = gy + row_idx * cell_h + cell_h // 2
                cv2.circle(dbg, (cx, cy), min(cell_w, cell_h) // 3, (0, 255, 0), -1)
                cv2.putText(dbg, resp, (cx - 8, cy + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        debug_path = str(Path(image_path).stem) + "_debug.jpg"
        cv2.imwrite(debug_path, dbg)
        print(f"[DEBUG] Imagem de depuração salva: {debug_path}")

    result = {"respostas": respostas}

    # 6. Calcular pontuação se gabarito fornecido
    if gabarito:
        acertos = 0
        erros = []
        for q_num, resp_aluno in respostas.items():
            if q_num <= len(gabarito):
                correta = gabarito[q_num - 1].upper()
                if resp_aluno == correta:
                    acertos += 1
                else:
                    erros.append({
                        "questao": q_num,
                        "resposta_aluno": resp_aluno,
                        "resposta_correta": correta
                    })
        result["acertos"] = acertos
        result["total"] = min(NUM_QUESTOES, len(gabarito))
        result["erros"] = erros
        result["nota"] = round(acertos / result["total"] * 10, 2)

    if debug_path:
        result["debug_image"] = debug_path

    return result


def print_result(result: dict):
    """Exibe o resultado formatado no terminal."""
    respostas = result["respostas"]

    print("\n" + "=" * 52)
    print("  GABARITO LIDO - CARTÃO-RESPOSTA OBMEP")
    print("=" * 52)
    print(f"  {'Q':>3}  {'Resp':^6}  {'Q':>3}  {'Resp':^6}  {'Q':>3}  {'Resp':^6}  {'Q':>3}  {'Resp':^6}")
    print("-" * 52)
    for i in range(0, NUM_QUESTOES, 4):
        linha = ""
        for j in range(4):
            q = i + j + 1
            if q <= NUM_QUESTOES:
                resp = respostas.get(q, "?")
                linha += f"  {q:>3}  {resp:^6}"
        print(linha)
    print("=" * 52)

    if "acertos" in result:
        print(f"\n  Acertos : {result['acertos']} / {result['total']}")
        print(f"  Nota    : {result['nota']:.1f} / 10.0")
        if result["erros"]:
            print(f"\n  Questões erradas:")
            for e in result["erros"]:
                print(f"    Q{e['questao']:>2}: marcou {e['resposta_aluno']}  |  correto: {e['resposta_correta']}")
        else:
            print("\n  Parabéns! Todas as respostas estão corretas!")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Lê gabarito de cartão-resposta OBMEP a partir de uma foto."
    )
    parser.add_argument("imagem", help="Caminho para a foto do cartão-resposta (JPG/PNG)")
    parser.add_argument(
        "--gabarito", "-g",
        help="Gabarito correto, ex: ABCDEABCDEABCDEABCDE (20 letras A-E)",
        default=None
    )
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Salva imagem de depuração com anotações visuais"
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Exibe resultado em formato JSON"
    )
    args = parser.parse_args()

    gabarito_lista = None
    if args.gabarito:
        gabarito_lista = list(args.gabarito.upper().replace(" ", ""))
        if len(gabarito_lista) != NUM_QUESTOES:
            print(f"[ERRO] O gabarito deve ter exatamente {NUM_QUESTOES} letras.")
            sys.exit(1)
        validas = set(ALTERNATIVAS)
        invalidas = [c for c in gabarito_lista if c not in validas]
        if invalidas:
            print(f"[ERRO] Letras inválidas no gabarito: {invalidas}. Use apenas A, B, C, D ou E.")
            sys.exit(1)

    result = read_answers(args.imagem, gabarito=gabarito_lista, debug=args.debug)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_result(result)


if __name__ == "__main__":
    main()
