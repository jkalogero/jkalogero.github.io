const publicationsData = [{
        title: "Scale Equivariant Graph Metanetworks",
        authors: "Ioannis Kalogeropoulos*, Giorgos Bouritsas*, Yannis Panagakis",
        venue: "Neural Information Processing Systems, NeurIPS 2024 (Oral)",
        link: "https://arxiv.org/pdf/2406.10685",
        code: "https://github.com/jkalogero/scalegmn",
        page: "/projects/scalegmn",
        bibtext: "@article{kalogeropoulos2024scale,\n" + "title={Scale Equivariant Graph Metanetworks},\n" + "author={Kalogeropoulos, Ioannis and Bouritsas, Giorgos and Panagakis, Yannis},\n" + "journal={arXiv preprint arXiv:2406.10685},\n" + "year={2024}\n" + "}"
    }, {
        title: "MLOps meets edge computing: an edge platform with embedded intelligence towards 6G systems",
        authors: "Nikos Psaromanolakis, Vasileios Theodorou, Dimitris Laskaratos, Ioannis Kalogeropoulos, Maria-Eleftheria Vlontzou, Eleni Zarogianni, Georgios Samaras",
        venue: "2023 Joint European Conference on Networks and Communications & 6G Summit",
        link: "https://ieeexplore.ieee.org/abstract/document/10188244/",
        bibtext: "@inproceedings{psaromanolakis2023mlops,\n" + "title={MLOps meets edge computing: an edge platform with embedded intelligence towards 6G systems},\n" + "author={Psaromanolakis, Nikos and Theodorou, Vasileios and Laskaratos, Dimitris and Kalogeropoulos, Ioannis and Vlontzou, Maria-Eleftheria and Zarogianni, Eleni and Samaras, Georgios},\n" + "booktitle={2023 Joint European Conference on Networks and Communications \\& 6G Summit (EuCNC/6G Summit)},\n" + "pages={496--501}\n" + "year={2023}\n" + "organization={IEEE}\n" + "}"
    }, {
        title: "Edgeds: Data spaces enabled multi-access edge computing",
        authors: "Ioannis Kalogeropoulos, Maria Eleftheria Vlontzou, Nikos Psaromanolakis, Eleni Zarogianni, Vasileios Theodorou",
        venue: "2023 Joint European Conference on Networks and Communications & 6G Summit",
        link: "https://ieeexplore.ieee.org/abstract/document/10188334/",
        bibtext: "@article{kalogeropoulos2023edgeds,\n" + "title={Edgeds: Data spaces enabled multi-access edge computing},\n" + "author={Kalogeropoulos, Ioannis and Vlontzou, Maria Eleftheria and Psaromanolakis, Nikos and Zarogianni, Eleni and Theodorou, Vasileios},\n" + "booktitle={2023 Joint European Conference on Networks and Communications \\& 6G Summit (EuCNC/6G Summit)},\n" + "pages={424--529}\n" + "year={2023}\n" + "organization={IEEE}\n" + "}"
    }
];

// Function to load the publications template and render the data
function loadPublications() {
    fetch('publications.html')
        .then(response => response.text())
        .then(template => {
            const publicationsContainer = document.getElementById("publications_section");

            publicationsData.forEach(pub => {
                const pubElement = document.createElement('div');

                const authorToUnderline = "Ioannis Kalogeropoulos";

                const strToItalic = "(Oral)";

                const underlinedAuthors = pub.authors.replace(authorToUnderline, `<u>${authorToUnderline}</u>`);
                const italicString = pub.venue.replace(strToItalic, `<strong>${strToItalic}</strong>`);

                pubElement.innerHTML = template;
                pubElement.querySelector('.publication-title').textContent = pub.title;

                pubElement.querySelector('.publication-authors').innerHTML = underlinedAuthors;
                // pubElement.querySelector('.publication-venue').textContent = pub.venue;
                pubElement.querySelector('.publication-venue').innerHTML = italicString;
                pubElement.querySelector('.publication-link').href = pub.link;
                pubElement.querySelector('.pup_bibtext').textContent = pub.bibtext;

                if (pub.code) {
                    pubElement.querySelector('.publication-code').href = pub.code;
                }
                else {
                    pubElement.querySelector('.publication-code').style.display = 'none';
                    pubElement.querySelector('#code-sep').style.display = 'none';
                }
                if (pub.page) {
                    pubElement.querySelector('.publication-page').href = pub.page;
                }
                else {
                    pubElement.querySelector('.publication-page').style.display = 'none';
                    pubElement.querySelector('#page-sep').style.display = 'none';
                }

                publicationsContainer.appendChild(pubElement);
            });
        })
}

// Load the navbar when the page loads
// window.onload = loadPublications();
// window.onload = function() {
//     loadNavbar();
//     loadPublications();
// };

function toggleBibtex(button) {
    var bibtexDiv = button.parentNode.parentNode.nextElementSibling
    // console.log(bibtexDiv)
    if (bibtexDiv.style.display === 'none') {
        bibtexDiv.style.display = 'block';
    } else {
        bibtexDiv.style.display = 'none';
    }
}

function copyBibtex(button) {
    var content = button.previousElementSibling.textContent;
    navigator.clipboard.writeText(content).then(function () {
        var originalText = button.textContent;
        button.textContent = 'Copied';

        // Set a timeout to revert the button text back to "Copy" after 3 seconds
        setTimeout(function () {
            button.textContent = originalText;
        }, 3000);


    }, function (err) {
        console.error('Could not copy BibTeX: ', err);
    });
}

function loadNavbar() {
    fetch('/navbar.html')
        .then(response => response.text())
        .then(data => {
            document.getElementById('navbar').innerHTML = data;
        });
}

function loadFooter() {
    fetch('/footer.html')
        .then(response => response.text())
        .then(data => {
            document.getElementById('footer').innerHTML = data;
        });
}