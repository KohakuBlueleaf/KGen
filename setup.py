from setuptools import setup, find_packages


setup(
    name="kgen",
    packages=find_packages(),
    version="0.0.1",
    license="Apache 2.0",
    url="",
    description=(
        "TIPO: Text to Image genration through "
        "text Presampling with LLMs for Optimal prompting"
    ),
    author="Shih-Ying Yeh(KohakuBlueLeaf)",
    author_email="apolloyeh0123@gmail.com",
    zip_safe=False,
    install_requires=[
        "transformers",
        "huggingface_hub",
        "torch",
    ],
    package_data={"": ["**/*.txt"]},
)
