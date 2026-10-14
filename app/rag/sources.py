"""URLs ingested into the knowledge base.

The global site (getnet.net/en) is mostly institutional, so the Brazilian product site and
its help centre (where the products in the test scenarios actually live) are included.
"""

GETNET_URLS: list[str] = [
    # Global / institutional (suggested source)
    "https://www.getnet.net/en",
    "https://www.getnet.net/en/our-solutions/in-person-payments",
    "https://www.getnet.net/en/our-solutions/online-payments",
    "https://www.getnet.net/en/our-solutions/omnichannel",
    "https://www.getnet.net/en/our-solutions/value-added-solutions",
    "https://www.getnet.net/en/corporate/about-us",
    # Brazil - products
    "https://site.getnet.com.br/",
    "https://site.getnet.com.br/todas-as-maquininhas/",
    "https://site.getnet.com.br/get-tap/",
    "https://site.getnet.com.br/pix/",
    "https://site.getnet.com.br/link-de-pagamento/",
    "https://site.getnet.com.br/crediario/",
    "https://site.getnet.com.br/conta-digital/",
    "https://site.getnet.com.br/getcheckout/",
    "https://site.getnet.com.br/emprestimo/",
    # Brazil - help centre (FAQ)
    "https://site.getnet.com.br/duvidas/",
    "https://site.getnet.com.br/duvidas/antecipacao-de-vendas/",
    "https://site.getnet.com.br/duvidas/pix/",
    "https://site.getnet.com.br/duvidas/pix-mais-respostas/",
    "https://site.getnet.com.br/duvidas/link-de-pagamento/",
    "https://site.getnet.com.br/duvidas/crediario/",
    "https://site.getnet.com.br/duvidas/get-conta/",
    "https://site.getnet.com.br/duvidas/get-smart/",
    "https://site.getnet.com.br/duvidas/get-tap/",
    "https://site.getnet.com.br/duvidas/algar/",
    "https://site.getnet.com.br/duvidas/operacional/",
    "https://site.getnet.com.br/duvidas/taxas/",
    "https://site.getnet.com.br/duvidas/parcelado/",
    "https://site.getnet.com.br/duvidas/receba-ja-2-dias/",
    "https://site.getnet.com.br/duvidas/aluguel/",
]
