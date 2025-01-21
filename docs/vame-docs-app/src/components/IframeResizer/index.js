import React, { useRef, useEffect } from 'react';

const IframeResizer = ({ src, heightBuffer = 20, maxCellHeight = 400, ...props }) => {
  const iframeRef = useRef(null);

  useEffect(() => {
    const injectCustomStyles = () => {
      const iframe = iframeRef.current;
      if (iframe) {
        try {
          const iframeDocument = iframe.contentDocument || iframe.contentWindow.document;

          // Inject custom CSS for Markdown cells and other notebook elements
          const style = iframeDocument.createElement('style');
          style.innerHTML = `
            /* Apply Docusaurus font style */
            html {
              background-color: var(--ifm-background-color);
              color: var(--ifm-font-color-base);
              color-scheme: var(--ifm-color-scheme);
              font: var(--ifm-font-size-base) / var(--ifm-line-height-base) var(--ifm-font-family-base);
              -webkit-font-smoothing: antialiased;
              -webkit-tap-highlight-color: transparent;
              text-rendering: optimizelegibility;
              text-size-adjust: 100%;
            }

            body {
              margin: 0;
              padding: 0;
            }

            /* Markdown cell styling */
            .jp-MarkdownOutput {
              margin: 0rem 0;
              padding: 0rem;
            }

            /* Ensure long outputs in other cells scroll vertically */
            .jp-OutputArea, .jp-Cell-outputArea {
              max-height: ${maxCellHeight}px;
              overflow-y: auto;
            }

            .jp-RenderedText, .jp-OutputArea-output {
              word-wrap: break-word;
              white-space: pre-wrap;
            }
          `;
          iframeDocument.head.appendChild(style);

          // Adjust iframe height after styles are applied
          const height = iframeDocument.body.scrollHeight + heightBuffer;
          iframe.style.height = `${height}px`;
        } catch (error) {
          console.warn('Could not access iframe content due to cross-origin restrictions.', error);
        }
      }
    };

    const iframe = iframeRef.current;
    if (iframe) {
      iframe.addEventListener('load', injectCustomStyles);
    }

    return () => {
      if (iframe) {
        iframe.removeEventListener('load', injectCustomStyles);
      }
    };
  }, [heightBuffer, maxCellHeight]);

  return (
    <iframe
      ref={iframeRef}
      src={src}
      style={{ width: '100%', border: 0, overflow: 'hidden' }}
      {...props}
    ></iframe>
  );
};

export default IframeResizer;
