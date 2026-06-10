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
FILL_THRESHOLD = 0.12


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
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


def _try_find_quad(contours, img_area, min_area_frac=0.10):
    """Tenta extrair um quadrilátero válido da lista de contornos."""
    for c in contours[:15]:
        peri = cv2.arcLength(c, True)
        for eps in [0.02, 0.03, 0.04, 0.05, 0.07]:
            approx = cv2.approxPolyDP(c, eps * peri, True)
            if len(approx) == 4:
                area = cv2.contourArea(approx)
                x, y, cw, ch = cv2.boundingRect(approx)
                aspect = max(cw, ch) / max(min(cw, ch), 1)
                if area > img_area * min_area_frac and aspect < 4.0:
                    return approx.reshape(4, 2).astype("float32")
    return None


def refine_card_warp(warped, warped_gray):
    """
    Passo 2 de retificação: encontra a borda precisa do cartão dentro do warp grosseiro
    e aplica uma segunda transformação de perspectiva para recortar apenas o cartão.
    """
    h, w = warped_gray.shape
    img_area = h * w

    blurred = cv2.GaussianBlur(warped_gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 40, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    for c in cnts[:10]:
        area = cv2.contourArea(c)
        if area < img_area * 0.30:
            continue
        peri = cv2.arcLength(c, True)
        for eps in [0.01, 0.02, 0.03, 0.04]:
            approx = cv2.approxPolyDP(c, eps * peri, True)
            if len(approx) == 4:
                x, y, cw, ch = cv2.boundingRect(approx)
                aspect = max(cw, ch) / max(min(cw, ch), 1)
                if 1.1 < aspect < 2.5:
                    refined = four_point_transform(warped, approx.reshape(4, 2).astype("float32"))
                    rh, rw = refined.shape[:2]
                    if 1.1 < rw / rh < 2.5:
                        return refined
    return warped


def detect_card_contour(gray):
    """
    Detecta o contorno retangular do cartão com múltiplas estratégias:
    1. Edge detection (Canny) com vários parâmetros
    2. Segmentação por brilho (cartão branco em fundo escuro)
    3. Bounding rect do maior blob branco como fallback
    """
    h, w = gray.shape
    img_area = h * w

    # ── Estratégia 1: Canny com múltiplos parâmetros ─────────────────────────
    for blur_k in [5, 7, 9]:
        for low, high in [(20, 60), (30, 100), (50, 150), (10, 40)]:
            blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
            edged = cv2.Canny(blurred, low, high)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
            cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            result = _try_find_quad(cnts, img_area)
            if result is not None:
                return result

    # ── Estratégia 2: Segmentação por brilho ─────────────────────────────────
    for thresh_val in [210, 190, 170, 150]:
        _, bright = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
        k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, k_close)
        bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, k_open)
        cnts, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        result = _try_find_quad(cnts, img_area, min_area_frac=0.08)
        if result is not None:
            return result
        # Fallback: bounding rect do maior blob
        for c in cnts[:3]:
            if cv2.contourArea(c) < img_area * 0.08:
                continue
            x, y, cw, ch = cv2.boundingRect(c)
            aspect = max(cw, ch) / max(min(cw, ch), 1)
            if aspect < 4.0:
                return np.array(
                    [[x, y], [x + cw, y], [x + cw, y + ch], [x, y + ch]],
                    dtype="float32"
                )

    return None


def find_answer_grid(warped_gray):
    """
    Localiza o grid de respostas (A-E × 1-20) dentro do cartão retificado,
    pulando a linha de cabeçalho com os números das questões.
    Retorna (x, y, w, h).

    Proporções calibradas para o cartão OBMEP retificado (warp + refinamento):
      - Linha de números (1-20): ~50–53% da altura do cartão
      - Linha A:  ~53–59%
      - Linha B:  ~59–64%
      - Linha C:  ~64–70%
      - Linha D:  ~70–75%
      - Linha E:  ~75–81%
    """
    h, w = warped_gray.shape
    answer_y_start = int(h * 0.53)   # logo após a linha de números
    answer_y_end   = int(h * 0.81)   # fim da linha E
    grid_x_start   = int(w * 0.06)   # pula labels "A B C D E" à esquerda
    grid_x_end     = int(w * 0.92)   # para antes dos labels à direita

    return (
        grid_x_start,
        answer_y_start,
        grid_x_end - grid_x_start,
        answer_y_end - answer_y_start,
    )


def analyze_circle(cell_gray):
    """
    Determina se um círculo está marcado (preenchido) e retorna (marcado, fill_ratio).
    Usa limiar fixo de 128 (meio-cinza) para detectar pixels escuros, evitando que
    o limiar adaptativo Otsu distorça a proporção em células individuais.
    """
    h, w = cell_gray.shape
    mx = int(w * 0.12)
    my = int(h * 0.12)
    region = cell_gray[my:h - my, mx:w - mx]
    if region.size == 0:
        return False, 0.0
    # Use a fixed threshold of 128 (mid-gray) to detect dark pixels
    dark_pixels = np.sum(region < 128)
    fill_ratio = dark_pixels / region.size
    return fill_ratio >= FILL_THRESHOLD, fill_ratio


def read_answers(image_path: str, gabarito: list = None, debug: bool = False) -> dict:
    """
    Lê o gabarito de uma imagem de cartão-resposta.

    Args:
        image_path: Caminho para a foto do cartão-resposta (JPG/PNG).
        gabarito:   Lista com as respostas corretas (ex: ["A","B",...]).
        debug:      Salva imagem de depuração com anotações visuais.

    Returns:
        dict com 'respostas', 'acertos'/'nota'/'erros' (se gabarito fornecido).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Não foi possível carregar a imagem: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Detectar e retificar o cartão (dois passos para maior precisão)
    card_pts = detect_card_contour(gray)
    if card_pts is not None:
        warped = four_point_transform(img, card_pts)
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        # Passo 2: refinar para recortar exatamente a borda do cartão
        warped = refine_card_warp(warped, warped_gray)
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        print(f"[INFO] Cartão retificado: {warped.shape[1]}×{warped.shape[0]}px")
    else:
        print("[AVISO] Borda do cartão não detectada. Usando imagem completa.")
        warped = img.copy()
        warped_gray = gray.copy()

    # 2. Localizar o grid de respostas
    gx, gy, gw, gh = find_answer_grid(warped_gray)
    grid_gray = warped_gray[gy:gy + gh, gx:gx + gw]

    if debug:
        dbg = warped.copy()
        cv2.rectangle(dbg, (gx, gy), (gx + gw, gy + gh), (0, 255, 0), 2)

    # 3. Dividir em células (5 linhas A-E × 20 colunas)
    cell_h = gh // len(ALTERNATIVAS)
    cell_w = gw // NUM_QUESTOES

    fill_matrix = np.zeros((len(ALTERNATIVAS), NUM_QUESTOES))

    for row_idx in range(len(ALTERNATIVAS)):
        for col_idx in range(NUM_QUESTOES):
            y1 = row_idx * cell_h
            y2 = y1 + cell_h
            x1 = col_idx * cell_w
            x2 = x1 + cell_w
            cell = grid_gray[y1:y2, x1:x2]
            _, ratio = analyze_circle(cell)
            fill_matrix[row_idx, col_idx] = ratio

            if debug:
                cx = gx + x1 + cell_w // 2
                cy = gy + y1 + cell_h // 2
                marked = ratio >= FILL_THRESHOLD
                color = (0, 0, 255) if marked else (200, 200, 200)
                cv2.circle(dbg, (cx, cy), min(cell_w, cell_h) // 3, color, 1)

    # 4. Selecionar resposta por maior preenchimento em cada coluna
    respostas = {}
    for col_idx in range(NUM_QUESTOES):
        col_fills = fill_matrix[:, col_idx]
        best_row = int(np.argmax(col_fills))
        best_fill = col_fills[best_row]
        second_fills = np.concatenate([col_fills[:best_row], col_fills[best_row + 1:]])
        second_best = float(np.max(second_fills)) if len(second_fills) > 0 else 0.0
        q = col_idx + 1
        # Mark as answered if:
        # 1. absolute fill is above minimum (some dark pixels present)
        # 2. clearly dominant over other options (relative dominance >= 1.7x)
        if best_fill >= 0.12 and (second_best == 0 or best_fill / second_best >= 1.7):
            respostas[q] = ALTERNATIVAS[best_row]
        else:
            respostas[q] = "?"

    # 5. Anotar respostas no debug
    debug_path = None
    if debug:
        for col_idx in range(NUM_QUESTOES):
            resp = respostas.get(col_idx + 1, "?")
            if resp != "?":
                row_idx = ALTERNATIVAS.index(resp)
                cx = gx + col_idx * cell_w + cell_w // 2
                cy = gy + row_idx * cell_h + cell_h // 2
                cv2.circle(dbg, (cx, cy), min(cell_w, cell_h) // 3, (0, 200, 0), -1)
                cv2.putText(dbg, resp, (cx - 8, cy + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        debug_path = str(Path(image_path).stem) + "_debug.jpg"
        cv2.imwrite(debug_path, dbg)
        print(f"[DEBUG] Imagem salva: {debug_path}")

    result = {"respostas": respostas}

    # 6. Pontuar contra gabarito
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
                        "resposta_correta": correta,
                    })
        total = min(NUM_QUESTOES, len(gabarito))
        result["acertos"] = acertos
        result["total"] = total
        result["erros"] = erros
        result["nota"] = round(acertos / total * 10, 2)

    if debug_path:
        result["debug_image"] = debug_path

    return result


def print_result(result: dict):
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
                linha += f"  {q:>3}  {respostas.get(q, '?'):^6}"
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
    parser.add_argument("imagem", help="Foto do cartão-resposta (JPG/PNG)")
    parser.add_argument("--gabarito", "-g",
                        help="Gabarito correto com 20 letras A-E, ex: ABCDEABCDEABCDEABCDE")
    parser.add_argument("--debug", "-d", action="store_true",
                        help="Salva imagem de depuração com anotações")
    parser.add_argument("--json", "-j", action="store_true",
                        help="Exibe resultado em JSON")
    args = parser.parse_args()

    gabarito_lista = None
    if args.gabarito:
        gabarito_lista = list(args.gabarito.upper().replace(" ", ""))
        if len(gabarito_lista) != NUM_QUESTOES:
            print(f"[ERRO] O gabarito deve ter exatamente {NUM_QUESTOES} letras.")
            sys.exit(1)
        invalidas = [c for c in gabarito_lista if c not in ALTERNATIVAS]
        if invalidas:
            print(f"[ERRO] Letras inválidas: {invalidas}. Use apenas A-E.")
            sys.exit(1)

    result = read_answers(args.imagem, gabarito=gabarito_lista, debug=args.debug)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_result(result)


if __name__ == "__main__":
    main()
