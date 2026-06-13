import React from 'react';
import {Redirect} from '@docusaurus/router';
import useBaseUrl from '@docusaurus/useBaseUrl';

// The site has no standalone landing page: the root redirects to the docs.
export default function Home() {
  return <Redirect to={useBaseUrl('/docs/intro')} />;
}
